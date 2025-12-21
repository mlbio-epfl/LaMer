from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from PIL import Image

from .prompt import get_webshop_prompt, get_webshop_prompt_short
from .memory import SimpleMemory
from .envs  import build_webshop_envs
from .projection import webshop_projection
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

class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, num_attempts, do_reflection, config):        
        ## MetaRL: num_attempts >= 2
        self.num_attempts = num_attempts
        self.num_processes = envs.num_processes
        ## memories of previous state and actions 
        self.memories = [SimpleMemory() for _ in range(self.num_attempts)] 
        ## reflections 
        self.reflections = [{} for _ in range(self.num_processes)]
        self.do_reflection = do_reflection
        self.reflection_type = config.env.get('reflection_type', 'reflection_only')
        assert self.reflection_type in ['history_and_reflection', 'reflection_only', 'history_only']
        ## curr_traj_idx is used to track the current trajectory index for MetaRL
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0 
        self.max_turns = config.env.get('max_turns', 20)
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
                
        ## reset memories, reflections and curr_traj_idx
        for memory in self.memories:
            memory.reset(self.num_processes)

        self.reflections = [{} for _ in range(self.num_processes)]
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0
        
        admissible_actions = [info['available_actions'] for info in infos]
        observations = {'text': self.build_text_obs(admissible_actions, phase='play'), 
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
    
    def restart(self):
        ''' Used for 2nd or N-th attempts '''
        obs, infos = self.envs.restart()
        self.curr_traj_idx += 1
        self.curr_turn_idx = 0 # [0 for _ in range(self.num_processes)]
        
        admissible_actions = [info['available_actions'] for info in infos]
        observations = {
            'text': self.build_text_obs(admissible_actions, phase='play'), 
            'image': None, 
            'anchor': obs
        }

        return observations, infos

    def step(self, text_actions: List[str], phase: str='play'):
        assert phase in ['play', 'reflect']
        if phase == 'reflect':
            # extract reflection from text_actions
            reflections, valids = self.projection_f(text_actions, phase='reflect')
            
            # self.reflections= reflections
            for i, reflection in enumerate(reflections):
                self.reflections[i][self.curr_traj_idx] = reflection

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
            actions, valids = self.projection_f(text_actions, phase='play')
            text_obs, rewards, dones, infos = self.envs.step(actions)
            
            # add action_valid to infos
            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])
            
            self.curr_turn_idx += 1
            text_obs = self.format_obs(text_obs)
            self.memories[self.curr_traj_idx].store({
                'text_obs': text_obs, 
                'action': actions, 
                'reward': rewards,
                'dones': dones,
                'won': [info['won'] for info in infos]
            })

            admissible_actions = [info['available_actions'] for info in infos]
            next_observations = {
                'text': self.build_text_obs(admissible_actions=admissible_actions, phase='play'),
                'image': None,
                'anchor': text_obs
            }


            rewards = to_numpy(rewards)
            dones = to_numpy(dones)

            return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, admissible_actions: List[List[str]]=None, phase: str = 'play') -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        assert phase in ['play', 'reflect']

        if self.curr_turn_idx == 0:
            curr_trajs = ['' for _ in range(self.num_processes)]
        else:
            history_length = 5 if phase == 'play' else 20
            curr_trajs, _ = self.memories[self.curr_traj_idx].fetch(history_length=history_length, max_to_show=15)

        past_trajs = [{} for _ in range(self.num_processes)]
        for traj_idx in range(self.curr_traj_idx):
            trajectories, _ = self.memories[traj_idx].fetch(history_length=3, max_to_show=6)
            for i in range(self.num_processes):
                past_trajs[i][traj_idx] = trajectories[i]
        
        for i in range(self.num_processes):
            if phase == 'play':
                available_actions = self.format_avail_actions(admissible_actions[i])
                reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)
            else:
                reformatted_available_actions = ''

            obs = get_webshop_prompt(
                phase,
                turn_idx=self.curr_turn_idx,
                traj_idx=self.curr_traj_idx,
                task_description=self.tasks[i],
                curr_traj=curr_trajs[i],
                past_traj=past_trajs[i],
                admissible_actions=reformatted_available_actions,
                reflection=self.reflections[i],
                reflection_type=self.reflection_type
            )

            if len(obs) > 20000:
                print(f"Warning len(obs)={len(obs)} is too long, in traj#{self.curr_traj_idx}, turn#{self.curr_turn_idx}, phase {phase}")
                obs = get_webshop_prompt_short(
                    phase,
                    turn_idx=self.curr_turn_idx,
                    traj_idx=self.curr_traj_idx,
                    task_description=self.tasks[i],
                    curr_traj=curr_trajs[i],
                    past_traj=past_trajs[i],
                    admissible_actions=reformatted_available_actions,
                    reflection=self.reflections[i],
                    reflection_type=self.reflection_type
                )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

def make_envs(config, val_only=False):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "webshop" in config.env.env_name.lower():
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs) # , resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs) # , resources_per_worker=resources_per_worker)

        num_attempts = config.env.get('num_attempts', 1)
        do_reflection = config.env.get('do_reflection', True)
        val_num_attempts = config.env.get('val_num_attempts', num_attempts)
        val_do_reflection = config.env.get('val_do_reflection', do_reflection)
        
        projection_f = partial(webshop_projection)
        
        if val_only:
            envs = None 
        else:
            envs = WebshopEnvironmentManager(_envs, projection_f, num_attempts, do_reflection, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, val_num_attempts, val_do_reflection, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs

    else:
        print("Environment not supported")
        exit(1)