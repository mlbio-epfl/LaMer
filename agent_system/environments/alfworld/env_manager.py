from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import json

from .prompt import get_alfworld_prompt
from .memory import SimpleMemory
from .envs  import build_alfworld_envs
from .projection import alfworld_projection
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

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def parse_tasktype(gamefile):
    task_typs = []
    for file in gamefile:
        traj_data = json.load(open(file.replace('game.tw-pddl', 'traj_data.json'), 'r'))
        task_typs.append(traj_data['task_type'])
    return task_typs
    
def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, num_attempts, do_reflection, config):
        ## MetaRL: num_attempts >= 2
        self.num_processes = envs.num_processes
        self.num_attempts = num_attempts
        ## init states
        self.init_states = [None for _ in range(self.num_processes)]
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
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        self.task_types = parse_tasktype(self.gamefile)
        for info, task_type in zip(infos, self.task_types):
            info['task_type'] = task_type

        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        ## initialize the history buffer
        for memory in self.memories:
            memory.reset(len(text_obs))    
        ## reset init states
        self.init_states = text_obs
        ## reflections 
        self.reflections = [{} for _ in range(self.num_processes)]
        ## curr_traj_idx is used to track the current trajectory index for MetaRL
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0 

        full_text_obs = self.build_text_obs(self.envs.get_admissible_commands, phase='play')
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def reflect(self):
        '''Get prompts for reflect phase.'''
        infos = [{
                "action_is_valid": True,
                "won": False,
                'task_type': self.task_types[i]
            } for i in range(self.num_processes)]
        
        observations = {
                'text': self.build_text_obs(phase='reflect'),
                'image': None,
                'anchor': ['reflection' for _ in self.build_text_obs(phase='reflect')]
            }

        return observations, infos

    def restart(self):
        ''' Used for 2nd or N-th attempts '''
        text_obs, image_obs, infos = self.envs.restart()
        self.curr_traj_idx += 1
        self.curr_turn_idx = 0
        full_text_obs = self.build_text_obs(self.envs.get_admissible_commands, phase='play')
        for info, task_type in zip(infos, self.task_types):
            info['task_type'] = task_type
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str], phase: str='play'):
        assert phase in ['play', 'reflect']
        if phase == 'play':
            actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands, phase='play')
            text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
            
            # add action_valid to infos
            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])
                info['task_type'] = self.task_types[i]
                
            self.memories[self.curr_traj_idx].store({
                                                    'text_obs': text_obs, 
                                                    'action': actions,
                                                    'reward': rewards,
                                                    'dones': dones,
                                                    'won': [info['won'] for info in infos]
                                                    })
            self.curr_turn_idx += 1 
            full_text_obs = self.build_text_obs(self.envs.get_admissible_commands, phase=phase)
            if infos[0].get("extra.gamefile") is None:
                infos = set_gamefile(infos, self.gamefile)

            next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
            rewards = to_numpy(rewards)
            dones = to_numpy(dones)

            return next_observations, rewards, dones, infos
        else:
            reflections, valids = self.projection_f(text_actions, phase='reflect')
            
            for i, reflection in enumerate(reflections):
                self.reflections[i][self.curr_traj_idx] = reflection

            # next_obs, rewards, dones, infos = self.envs.step(actions)
            infos = [{
                "action_is_valid": False,
                "won": False
            } for _ in range(self.num_processes)]

            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])
                info['task_type'] = self.task_types[i]
                
            next_observations = {
                    'text': '',  
                    'image': None, 
                    'anchor': '' 
                }
            rewards = np.array(valids)
            dones = np.array([False] * len(text_actions))
            return next_observations, rewards, dones, infos

    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, admissible_actions: List[List[str]]=None, phase='play') -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        assert phase in ['play', 'reflect']

        if self.curr_turn_idx == 0:
            curr_trajs = ['' for _ in range(self.num_processes)]
        else:
            curr_trajs, _ = self.memories[self.curr_traj_idx].fetch(history_length=15)

        past_trajs = [{} for _ in range(self.num_processes)]
        for traj_idx in range(self.curr_traj_idx):
            trajectories, _ = self.memories[traj_idx].fetch(history_length=15)
            for i in range(self.num_processes):
                past_trajs[i][traj_idx] = trajectories[i]

        for i in range(self.num_processes):
            # exclude 'help' in admissible_actions[i]
            if phase == 'play':
                reformatted_admissible_actions = ",\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')
            else:
                reformatted_admissible_actions = ""

            obs = get_alfworld_prompt(
                phase,
                turn_idx=self.curr_turn_idx,
                traj_idx=self.curr_traj_idx,
                init_observation=self.init_states[i],
                curr_traj=curr_trajs[i],
                past_traj=past_trajs[i],
                admissible_actions=reformatted_admissible_actions,
                reflection=self.reflections[i],
                reflection_type=self.reflection_type
            )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break
    
    
    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check info['won'] of the last step.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            # self._process_batch(bs, total_batch_list, total_infos, success)
            task_type = total_infos[bs][0]['task_type']
            wons = [False for _ in range(self.num_attempts)]
            for i in reversed(range(len(total_batch_list[bs]))):  
                batch_item = total_batch_list[bs][i] 
                if batch_item['active_masks']:
                    info = total_infos[bs][i]
                    traj_idx = batch_item['traj_idx']
                    if batch_item['phase'] == 'play':
                        wons[traj_idx] = wons[traj_idx] or info['won']

            _won = False            
            for traj_idx, won in enumerate(wons):      
                _won = _won or won
                success[f'{task_type}|success_rate[{traj_idx}]'].append(_won)
                success[f'success_rate[{traj_idx}]'].append(_won)
        
        return {key: np.array(value) for key, value in success.items()}


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "alfworld" in config.env.env_name.lower():
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.get('eval_dataset', 'eval_all'), # 'eval_in_distribution' or 'eval_out_of_distribution' or 'eval_all'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs)
        
        num_attempts = config.env.get('num_attempts', 1)
        do_reflection = config.env.get('do_reflection', True)
        val_num_attempts = config.env.get('val_num_attempts', num_attempts)
        val_do_reflection = config.env.get('val_do_reflection', True)

        projection_f = partial(alfworld_projection)
        
        envs = AlfWorldEnvironmentManager(_envs, projection_f, num_attempts, do_reflection, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, val_num_attempts, val_do_reflection, config)
        return envs, val_envs
 
    else:
        print("Environment not supported")
        exit(1)
