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

def webshop_projection(actions: List[str], phase='play'):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    """
    actions = copy.deepcopy(actions)
    
    if phase == 'play':
    
        valids = [0] * len(actions)
        plans = [''] * len(actions)

        for i in range(len(actions)):
            original_str = actions[i]  # keep the original string
            actions[i] = actions[i].lower()

            # Attempt to extract the substring within <action>...</action>
            start_tag = "<action>"
            end_tag = "</action>"
            start_idx = actions[i].find(start_tag)
            end_idx = actions[i].find(end_tag)
            try:
                if start_idx == -1 or end_idx == -1:
                    # If we can't find a valid <action>...</action> block, mark as invalid
                    actions[i] = actions[i][-20:]  # 0 is invalid action for Sokoban
                    continue

                # Extract just the content between the tags
                extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
                
                actions[i] = extracted_action
                valids[i] = 1

            except:
                # randomly choose an action from the action list if illegal
                actions[i] = actions[i][-20:]
                
        return actions, valids
    
    else:
        # reflect phase
        valids = [0] * len(actions)
        reflections = [''] * len(actions)

        for i in range(len(actions)):
            action = actions[i]
            start_tag = "<remark>"
            start_idx = action.rfind(start_tag)
            end_tag = "</remark>"
            end_idx = action.rfind(end_tag)
            if start_idx == -1 or end_idx == -1:
                reflections[i] = ''
            else:
                reflections[i] = action[start_idx + len(start_tag):end_idx].strip()[:2000] # max 2000 characters
                valids[i] = 1

        return reflections, valids