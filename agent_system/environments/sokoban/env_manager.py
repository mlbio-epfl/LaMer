from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from PIL import Image

from .prompt import get_sokoban_prompt
from .memory import SimpleMemorySokoban
from .envs  import build_sokoban_envs
from .projection import sokoban_projection
from ..base import EnvironmentManagerBase

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "still",
        1: "up",
        2: "down",
        3: "left",
        4: "right",
    }
    def __init__(self, envs, projection_f, num_attempts, do_reflection, config):
        ## MetaRL: num_attempts >= 2
        self.num_attempts = num_attempts
        self.num_processes = envs.num_processes
        '''Action grouping'''
        self.num_actions_per_turn = config.env.get('num_actions_per_turn', 3)
        ## init states
        self.init_states = [None for _ in range(self.num_processes)]
        ## memories of previous state and actions
        self.memories = [SimpleMemorySokoban() for _ in range(self.num_attempts)] 
        ## reflections 
        self.reflections = [{} for _ in range(self.num_processes)]
        self.do_reflection = do_reflection
        self.reflection_type = config.env.get('reflection_type', 'reflection_only')
        assert self.reflection_type in ['history_and_reflection', 'reflection_only', 'history_only']
        ## curr_traj_idx is used to track the current trajectory index for MetaRL
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0 
        self.max_turns = config.env.get('max_turns', 7)
        super().__init__(envs, projection_f, config)

        ''' A normal pipeline should be:
        # Attempt 0, phase 'play'
        obs, info = envs.reset() 
        for step in range(max_steps):
            response = llm.generate(obs)
            obs, info = envs.step(response, phase='play')
        
        # Attempt 1, phase 'replect'
        obs, info = envs.reflect()
        response = llm.generate(obs)
        obs, info = envs.step(response, phase='reflect')
        
        # Attempt 1, phase 'play'
        obs, info = envs.restart() 
        for step in range(max_steps):
            response = llm.generate(obs)
            obs, info = envs.step(response, phase='play')
        ...
        '''

    def reset(self):
        obs, infos = self.envs.reset()
        ## reset init states
        self.init_states = obs
        ## reset memories, reflections and curr_traj_idx
        for memory in self.memories:
            memory.reset(self.num_processes)

        self.reflections = [{} for _ in range(self.num_processes)]
        self.curr_turn_idx = 0 #[0 for _ in range(self.num_processes)]
        self.curr_traj_idx = 0

        self.pre_text_obs = obs
        observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': obs
        }
        return observations, infos

    def restart(self):
        ''' Used for 2nd or N-th attempts '''
        obs, infos = self.envs.restart()
        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0 
        
        self.pre_text_obs = obs
        observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': obs
        }

        return observations, infos
    
    def reflect(self):
        '''Get prompts for reflect phase.'''
        infos = [{
                "action_is_valid": True,
                "won": False
            } for _ in range(self.num_processes)]
        
        observations = {
                'text': self.build_text_obs(phase='reflect'),
                'image': None,
                'anchor': ['reflection' for _ in self.build_text_obs(phase='reflect')]
            }

        return observations, infos
        
    def step(self, text_actions: List[str], phase: str='play'):
        assert phase in ['play', 'reflect']
        if phase == 'reflect':
            # extract reflection from text_actions
            reflections, valids = self.projection_f(text_actions, phase='reflect', num_actions_per_turn=self.num_actions_per_turn)
            for i, reflection in enumerate(reflections):
                self.reflections[i][self.curr_traj_idx] = reflection

            # next_obs, rewards, dones, infos = self.envs.step(actions)
            infos = [{
                "action_is_valid": False,
                "won": False
            } for _ in range(self.num_processes)]

            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])
                
            next_observations = {
                    'text': '',  
                    'image': None, 
                    'anchor': '' 
                }
            rewards = np.array(valids)
            dones = np.array([False] * len(text_actions))
            return next_observations, rewards, dones, infos

        else: 
            thoughts, actions, valids = self.projection_f(text_actions, phase='play', num_actions_per_turn=self.num_actions_per_turn)
            next_obs, rewards, dones, infos = self.envs.step(actions)

            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])

            self.memories[self.curr_traj_idx].store({'text_obs': next_obs, 
                                                     'thought': thoughts,
                                                    'action': [[self.ACTION_LOOKUP[act] for act in acts] for acts in actions], 
                                                    'reward': rewards,
                                                    'dones': dones,
                                                    'won': [info['won'] for info in infos]})
            self.pre_text_obs = next_obs
            self.curr_turn_idx += 1
            next_observations = {
                'text': self.build_text_obs(phase=phase),  
                'image': None, 
                'anchor': next_obs
            }
            
            rewards = to_numpy(rewards)
            dones = to_numpy(dones)
    
            return next_observations, rewards, dones, infos

    def build_text_obs(self, phase: str = 'play') -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        assert phase in ['play', 'reflect']

        obs_length = 2 if phase == 'play' else 7
    
        if self.curr_turn_idx == 0:
            curr_trajs = ['' for _ in range(self.num_processes)]
        else:
            curr_trajs, _ = self.memories[self.curr_traj_idx].fetch(obs_length=obs_length)

        past_trajs = [{} for _ in range(self.num_processes)]
        for traj_idx in range(self.curr_traj_idx):
            trajectories, _ = self.memories[traj_idx].fetch()
            for i in range(self.num_processes):
                past_trajs[i][traj_idx] = trajectories[i]
        
        for i in range(self.num_processes):
            obs = get_sokoban_prompt(phase=phase,
                                    turn_idx=self.curr_turn_idx,
                                    traj_idx=self.curr_traj_idx,
                                    init_observation=self.init_states[i],
                                    curr_traj=curr_trajs[i],
                                    past_traj=past_trajs[i],
                                    reflection=self.reflections[i],
                                    num_actions_per_turn=3,
                                    reflection_type=self.reflection_type,
                    )
                    
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "sokoban" in config.env.env_name.lower():
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth,
            'min_steps': config.env.get('min_steps', 5),  # default to 3 if not specified
            'max_sol_steps': config.env.get('max_sol_steps', config.env.max_steps) 
            }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs)
        
        num_attempts = config.env.get('num_attempts', 1)
        do_reflection = config.env.get('do_reflection', True)
        val_num_attempts = config.env.get('val_num_attempts', num_attempts)
        val_do_reflection = config.env.get('val_do_reflection', do_reflection)

        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, num_attempts, do_reflection, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, val_num_attempts, val_do_reflection, config)
        return envs, val_envs
 
    else:
        print("Environment not supported")
        exit(1)