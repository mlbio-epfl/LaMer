import torch
import random
from typing import List
import re
import copy
import json

def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False
    
def sokoban_projection(actions: List[str], phase='play', num_actions_per_turn=3):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Sokoban action mappings:
    - 0: Still (Invalid Action)
    - 1: Up
    - 2: Down
    - 3: Left
    - 4: Right
    """
    _actions = actions
    actions = copy.deepcopy(actions)
    if phase == 'play':
        action_pools = {
            "up": 1,
            "down": 2,
            "left": 3,
            "right": 4,
            "still": 0,
        }

        valids = [0] * len(actions)
        plans = [''] * len(actions)

        for i in range(len(actions)):
            original_str = actions[i]  # keep the original string
            processed_actions = []

            # Attempt to extract the substring within <action>...</action>
            start_tag = "<action>"
            start_idx = actions[i].find(start_tag)
            end_tag = "</action>"
            end_idx = actions[i].rfind(end_tag)
            try:
                if start_idx == -1 or end_idx == -1:
                    # If we can't find a valid <action>...</action> block, mark as invalid
                    actions[i] = [0]  # 0 is invalid action for Sokoban
                    continue

                # Extract just the content between the tags
                extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
                for _ext_act in extracted_action.split(','):
                    for act in action_pools.keys():
                        if act in _ext_act:
                            processed_actions.append(action_pools[act])
                            # if found legal action, set valids to 1
                            break

                if len(processed_actions) > 0:
                    actions[i] = processed_actions[:num_actions_per_turn]
                    valids[i] = 1

                # If no valid action found, randomly select from pool
                if valids[i] == 0:
                    actions[i] = [0]

            except:
                # randomly choose an action from the action list if illegal
                actions[i] = [0]

            # check MEMORY_UPDATE
            plan_start_tag = "<plan>"
            plan_end_tag = "</plan>"

            plan_start_idx = original_str.rfind(plan_start_tag)
            plan_end_idx = original_str.rfind(plan_end_tag)
            if plan_start_idx == -1 or plan_end_idx == -1:
                plans[i] = ''
            else:
                plans[i] = original_str[plan_start_idx + len(plan_start_tag):plan_end_idx].strip()
        
        return plans, actions, valids
    else:
        # reflect phase
        valids = [0] * len(actions)
        reflections = [''] * len(actions)

        for i in range(len(actions)):
            action = actions[i]
            start_tag = "<remark>"
            start_idx = action.find(start_tag)
            end_tag = "</remark>"
            end_idx = action.find(end_tag)
            if start_idx == -1 or end_idx == -1:
                reflections[i] = ''
            else:
                reflections[i] = action[start_idx + len(start_tag):end_idx].strip()[:2000] # max 2000 characters
                valids[i] = 1

        return reflections, valids