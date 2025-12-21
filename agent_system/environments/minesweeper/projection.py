
import re
import random
import copy
from typing import List

def minesweeper_projection(actions: List[str], board_size: int, phase='play'):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    """
    actions = copy.deepcopy(actions)
    if phase == 'play':
        valids = [0] * len(actions)
        plans = [''] * len(actions)
        
        for i in range(len(actions)):
            original_str = actions[i]
            
            start_idx = actions[i].find("<action>")
            end_idx = actions[i].find("</action>")

            if start_idx == -1 or end_idx == -1:
                # random action
                # action = random.choice(["L", "M", "R"])
                # action = "L"
                # row_idx = str(random.randint(1, board_size))
                # col_idx = str(random.randint(1, board_size))
                # valids[i] = 0
                # actions[i] = (row_idx, col_idx)
                valids[i] = 0
                actions[i] = (-1, -1)
                continue

            extracted_action = actions[i][start_idx + len("<action>"):end_idx].strip()
            # row_idx, col_idx, valid = parse_action_str(extracted_action, board_size)
            match_result = re.search(r"\(( *\d+) *, *(\d+) *\)", extracted_action)
            try:
                row_idx, col_idx = match_result.groups()
                valid = 1
            except:
                valid = 0
                row_idx = -1 # str(random.randint(1, board_size))
                col_idx = -1 # str(random.randint(1, board_size))

            actions[i] = (row_idx, col_idx)
            valids[i] = valid
            
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
        
        # return actions, valids
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
    

# def parse_action_str(response: str, board_size: int):
#     # response_ = re.sub(r"(?s)(.*)ACTION:", "", response)
#     # response_ = response_.strip()
#     match_result = re.search(r"([LMR]) *\(( *\d+) *, *(\d+) *\)", response)
#     try:
#         action, row_idx, col_idx = match_result.groups()
#         valid = 1
#     except (AttributeError, TypeError):
#         action, row_idx, col_idx, valid = parse_action_str_in_natural_language(response, board_size)

#     return action, row_idx, col_idx, valid

# def parse_action_str_in_natural_language(response: str, board_size: int):
#     # response = re.findall(r"^(ACTION:.*(?:\r?\n|$))", response, re.MULTILINE)
#     # if len(response) == 0:
#     #     # Random fallbacks
#     #     action = random.choice(["L", "M", "R"])
#     #     row_idx = str(random.randint(1, board_size))
#     #     col_idx = str(random.randint(1, board_size))
#     #     valid = 0
#     #     return action, row_idx, col_idx, valid
    
#     # response = response[-1].strip()

#     try:
#         action = re.findall(r"(?:left|middle|right)[- ]?click(?:ing)?", response)[-1]
#         row_idx, col_idx = re.search(r"\(( *\d+) *, *(\d+) *\)", response).groups()
        
#         if "left" in action:
#             action = "L"
#         elif "middle" in action:
#             action = "M"
#         elif "right" in action:
#             action = "R"
        
#         valid = 1
#         return action, row_idx, col_idx, valid
        
#     except (AttributeError, TypeError, IndexError):
#         # Random fallbacks
#         # action = random.choice(["L", "M", "R"])
#         action = "L"
#         row_idx = str(random.randint(1, board_size))
#         col_idx = str(random.randint(1, board_size))
#         valid = 0
#         return action, row_idx, col_idx, valid