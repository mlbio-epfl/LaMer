from omegaconf import DictConfig, OmegaConf

import os
import numpy as np
import random
import ray
import json
import warnings
warnings.filterwarnings('ignore')

"""python -m examples.test_env
"""

env_name = 'sokoban'  # 'minesweeper' or 'sokoban' or 'webshop' or 'alfworld'
os.environ['ALFWORLD_DATA']='/your/alfworld/path' # only needed for alfworld

def create_envs(config):
    if env_name == 'sokoban':
        from agent_system.environments.sokoban import make_envs
    elif env_name == 'minesweeper':
        from agent_system.environments.minesweeper import make_envs
    elif env_name == 'alfworld':
        from agent_system.environments.alfworld import make_envs
    elif env_name == 'webshop':
        from agent_system.environments.webshop import make_envs
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    
    return make_envs(config)

def init_config() -> DictConfig:
    config = OmegaConf.load(f"verl/trainer/config/eval/{env_name}.yaml")
    return config

def random_action(obs_list, info_list):
    if env_name == 'sokoban':
        num_actions_per_turn = 3
        actions = []
        for _ in range(len(info_list)):
            action = ','.join([np.random.choice(["up", "down", "left", "right"]) for _ in range(num_actions_per_turn)])
            actions.append(f"<action>{action}</action>")
    elif env_name == 'minesweeper':
        actions = []
        for anchor_str in obs_list['anchor']:
            anchor_str = anchor_str.replace('Row 1: ', '').replace('Row 2: ', '').replace('Row 3: ', '').replace('Row 4: ', '').replace('Row 5: ', '').replace('Row 6: ', '')
            arr = [row.split() for row in anchor_str.split("\n")]
            question_marks = [(r + 1, c + 1)          # convert to 1–6 indexing
                          for r in range(6)
                          for c in range(6)
                          if arr[r][c] == '?']
            x, y = random.choice(question_marks)
            actions.append(f"<action>({x}, {y})</action>")
    elif env_name == 'alfworld':
        actions = ['<action>'+np.random.choice(_info['admissible_commands'])+'</action>' for _info in info_list]
    elif env_name == 'webshop':
        from agent_system.environments.webshop.webshop.web_agent_site.models.models import RandomPolicy
        policy = RandomPolicy()
        actions = []
        for _info in info_list:
            available_actions = _info['available_actions']
            action = policy.forward('', available_actions)
            actions.append(f"<action>{action}</action>")
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    
    return actions

def main():
    ray.init(log_to_driver=False)
    config = init_config()
    config.data.train_batch_size = 1
    config.data.val_batch_size = 1
    print(config.env)
    N = config.data.val_batch_size

    _, val_envs = create_envs(config)

    prompts = []
    obs_list, info_list = val_envs.reset()
    prompts.append('[Attempt 0]\n' + obs_list['text'][0])

    for traj_idx in range(3):
        if traj_idx >= 1:
            obs_list, info_list = val_envs.reflect()
            prompts.append('[Reflection]\n' + obs_list['text'][0])

            reflections = ['<remark>In my previous trial, I did ... I should have ...</remark>'] * N
            obs_list, reward_list, done_list, info_list = val_envs.step(reflections, phase='reflect')

            obs_list, info_list = val_envs.restart()
            prompts.append(f'[Attempt {traj_idx}]\n' + obs_list['text'][0])

        for _ in range(7):
            actions = random_action(obs_list, info_list)
            obs_list, reward_list, done_list, info_list = val_envs.step(actions, phase='play')
            prompts.append(f'[Attempt {traj_idx}]\n' + obs_list['text'][0])

            if np.all(done_list):
                break

    for prompt in prompts:
        print(prompt)

if __name__ == '__main__':
    main()