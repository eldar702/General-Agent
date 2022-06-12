
##############################            Imports & Globals              #################################
import copy
import random
import sys
import os
import time
import numpy as np
from pddlsim.executors.executor import Executor
from pddlsim.local_simulator import LocalSimulator
from pddlsim import parser_independent as parsr_i
import helpers.helper_functions as helper
from helpers import custom_parser_independent as cu_parsr

# read files:
input_flag = sys.argv[1]
domain_name = sys.argv[2]
problem_name = sys.argv[3]
policy_files_folder_path = os.getcwd() + ("/" + domain_name).replace(".pddl", "", 2)
policy_files_path = (os.getcwd() + ("/" + domain_name) + "/" + problem_name).replace(".pddl", "", 3)

# GLOBALS
TIMER = time.time()
FULL_LAST_ACTION = None
LAST_STATE_HASH, CURRENT_STATE_HASH = None, None
COUNTER = 0
DISTANCE_TO_CHECK = 10
FOUND_ALL_GOALS = False
#########################################################################################################
###########################            General Agent - Learner              #############################
#########################################################################################################

class GeneralLearnerAgent(Executor):
    ##########################             Init Functions               #################################
    def __init__(self):
        super(GeneralLearnerAgent, self).__init__()

        self.alpha, self.gamma, self.epsilon = 0.8, 0.6, 1
        self.actions_count, self.deterministic_act_dict, self.actions_probs = {}, {}, {}
        self.data = {}
        self.new_pddl = None
        self.max_reward, self.uncompleted_goals = 0, 0
        self.distance = 3   # stupid learn distance
        self.last_x_states = []
        self.is_deterministic = False
        self.state_machine = "stupid_learner"
        self.current_state, self.last_state, self.last_action = None, None, None

    def initialize(self, services):
        self.services = services
        self.make_actions_deterministic()
        self.read_policies_files()
        self.create_deterministic_domain()
        self.new_pddl = cu_parsr.PDDL(self.services.parser)
        self.max_reward = self.uncompleted_goals = self.get_uncompleted_goals(
            self.services.goal_tracking.uncompleted_goals[0])

        pass
        ##########################              Run Agent Run!                  #################################

    def next_action(self):
        self.actions_in_new_round()

        if self.services.goal_tracking.reached_all_goals() and helper.minute_passed(0.1, TIMER):
            return None
        chosen_action = self.state_machines()
        self.actions_in_end_round(chosen_action)
        return chosen_action
        pass

      ##########################              General Func                  ###################
    def state_machines(self):
        global LAST_STATE_HASH, CURRENT_STATE_HASH, COUNTER
        valid_actions = self.services.valid_actions.get()
        self.feel_data(CURRENT_STATE_HASH, valid_actions)
        if len(valid_actions) == 0: return None
        elif len(valid_actions) == 1:
            chosen_action = self.services.valid_actions.get()[0]
        if self.state_machine == "stupid_learner":
            if COUNTER % 3:
                chosen_action = self.choose_curiosity_random(valid_actions)
            else:
                chosen_action = random.choice(valid_actions)

        elif self.state_machine == "smart_learner":
            pass
        return chosen_action

    def actions_in_new_round(self):
        global LAST_STATE_HASH, CURRENT_STATE_HASH
        self.current_state = self.services.perception.get_state()
        LAST_STATE_HASH, CURRENT_STATE_HASH = helper.make_hash_sha256(self.last_state), helper.make_hash_sha256(self.current_state)
        self.change_last_x_states()
        self.update_Q_table()
        self.update_RMax_dict()
        self.write_policies_files()

    def actions_in_end_round(self, chosen_action):
        global FULL_LAST_ACTION, COUNTER
        self.last_state = copy.deepcopy(self.current_state)
        FULL_LAST_ACTION = chosen_action
        self.last_action = chosen_action.split()[0].split('(')[1]
        COUNTER += 1
        pass
    
    def write_policies_files(self):
        global policy_files_path
        helper.is_directory_exists(policy_files_folder_path)
        helper.save_dict_to_file(self.data, policy_files_path + "-Q-learning")  # save the Q-learning table as file
        helper.save_dict_to_file(self.actions_count, policy_files_path + "-Rmax-count")  # save the actions count
        helper.save_dict_to_file(self.actions_probs, policy_files_path + "-Rmax-prob")  # save the actions probabilities

        ####################             Q - LEARNING  Methods               #############################

    #######################             R-Max  TABLE  Methods               ################################
    def init_RMax_dict(self):
        self.actions_count = helper.init_dict(self.deterministic_act_dict, 0)
        self.actions_probs = copy.deepcopy(self.actions_count)

    def update_RMax_dict(self):
        if self.last_action is None: return

        idx_of_happened_act = self.check_which_act_happened()
        if idx_of_happened_act is not None:
            self.actions_count[self.last_action][idx_of_happened_act] += 1
        self.calculate_probabilities()

    def check_which_act_happened(self):
        global FULL_LAST_ACTION
        if self.last_action is None: return
        hash_current_state = helper.make_hash_sha256(self.current_state)
        for idx, action in self.deterministic_act_dict[self.last_action].items():
            if action == "stay_in_place":
                hash_temp_state = helper.make_hash_sha256(self.last_state)
                if hash_current_state == hash_temp_state: return idx
            else:
                temp_state = copy.deepcopy(self.last_state)
                cu_parsr.PDDL.apply_action_to_state(self.new_pddl, FULL_LAST_ACTION, temp_state, action,  check_preconditions=False)
                hash_temp_state = helper.make_hash_sha256(temp_state)
                if hash_current_state == hash_temp_state: return idx
        pass

    def calculate_probabilities(self):
        for action in self.actions_count.items():
            action_name = action[0]
            prob, count = 0, 0
            for value in action[1].values():
                count += value
            for idx, val in action[1].items():
                if count != 0:
                    self.actions_probs[action_name][idx] = "{:.3f}".format(float(val) / float(count))
                else:
                    self.actions_probs[action_name][idx] = 0

    def choose_curiosity_random(self, valid_actions):
        best_action = []
        for action in valid_actions:
            action_name = action.split()[0].split('(')[1]
            for deterministic_act in self.deterministic_act_dict[action_name].values():
                if deterministic_act == "stay_in_place": continue
                temp_state = copy.deepcopy(self.current_state)
                cu_parsr.PDDL.apply_action_to_state(self.new_pddl, action, temp_state, deterministic_act, check_preconditions=False)
                hash_state = helper.make_hash_sha256(temp_state)
                if hash_state not in self.data.keys() or self.data[hash_state]["visited"] == 0:
                    if action not in best_action:
                        best_action.append(action)
        if len(best_action) == 0:
            return random.choice(valid_actions)
        else:
            return random.choice(best_action)
    #######################             Q-table  Methods               ################################
    def update_Q_table(self):
        global LAST_STATE_HASH, FOUND_ALL_GOALS

        if self.last_action is None: return
        if LAST_STATE_HASH not in self.data.keys() or FOUND_ALL_GOALS is True:
            self.create_new_state_action_data(LAST_STATE_HASH, action=self.last_action)
        self.data[LAST_STATE_HASH]["visited"] = 1
        self.data[LAST_STATE_HASH]["last_action"] = self.last_action
        # get inside ONLY the first time we complete all goals
        if len(self.services.goal_tracking.uncompleted_goals) == 0:
            FOUND_ALL_GOALS = True
            self.check_if_found_goal(0)
            reward = DISTANCE_TO_CHECK + 1
        else:
            uncompleted_goals_number = self.get_uncompleted_goals(self.services.goal_tracking.uncompleted_goals[0])
            self.check_if_found_goal(uncompleted_goals_number)
            reward = self.get_reward(LAST_STATE_HASH)

        self.data[LAST_STATE_HASH]["actions"][self.last_action] = round(((1 - self.alpha) * self.data[LAST_STATE_HASH]["actions"][self.last_action])\
                                                             + (self.alpha * (reward + self.gamma * np.max(self.data[LAST_STATE_HASH]["actions"].values()))),3)

    def create_new_state_action_data(self, key, action=None, visited=1):
        if action is not None:
            self.data[key] = {'actions': {action: 0}, 'dist_from_goal': 0, "is_goal": False, 'visited': visited, "last_action": action}
        else:
            self.data[key] = {'actions': {}, 'dist_from_goal': 0, "is_goal": False, 'visited': visited, "last_action": None}

    ######################             Reward Functions               ##############################
    def change_dist_to_goal(self):
        for idx, state_hash in enumerate(self.last_x_states):
            if self.data[state_hash]['dist_from_goal'] > 0:
                self.data[state_hash]['dist_from_goal'] = min((self.distance - idx), self.data[state_hash]['dist_from_goal'])
            else:  self.data[state_hash]['dist_from_goal'] = (self.distance - idx)


    def check_if_found_goal(self, uncompleted_goals_num=0):
        global LAST_STATE_HASH
        if uncompleted_goals_num < self.uncompleted_goals:
            self.uncompleted_goals = uncompleted_goals_num
            self.data[LAST_STATE_HASH]["is_goal"] = True
            self.change_dist_to_goal()
            self.change_reward_by_distance()
            return True
        elif uncompleted_goals_num > self.uncompleted_goals:
            self.uncompleted_goals = uncompleted_goals_num
            self.data[LAST_STATE_HASH]["actions"][self.last_action] = -20
        return False

    def change_reward_by_distance(self):
        for state_hash in self.last_x_states:
            reward = self.get_reward(state_hash)
            last_action = self.data[state_hash]["last_action"]
            self.data[state_hash]["actions"][last_action] = round(
                ((1 - self.alpha) * self.data[state_hash]["actions"][last_action]) \
                + (self.alpha * (reward + self.gamma * np.max(self.data[state_hash]["actions"].values()))), 3)

    def get_reward(self, checked_state_hash):
        if self.data[checked_state_hash]["is_goal"]:
            return self.distance + 2
        elif self.data[checked_state_hash]["dist_from_goal"] != 0:
            return self.distance - self.data[checked_state_hash]["dist_from_goal"] + 1
        else: return -1

    def get_uncompleted_goals(self, goal, reward=0):
        if isinstance(goal, parsr_i.Literal):
            return 1
        if isinstance(goal, parsr_i.Conjunction):
            for part_goal in goal.parts:
                reward += self.get_uncompleted_goals(part_goal)
        elif isinstance(goal, parsr_i.Disjunction):
            max_reward = 0
            for part_goal in goal.parts:
                temp_reward = self.get_uncompleted_goals(part_goal)
                if temp_reward > max_reward:
                    max_reward = temp_reward
            reward += max_reward
        return reward

    #############################              PDDL Functions              ###################################
    def read_policies_files(self):
        global policy_files_path
        rmax_prob, rmax_count = (policy_files_path + "-Rmax-prob"), (policy_files_path + "-Rmax-count")
        q_learning = policy_files_path + "-Q-learning"

        if helper.is_file_exists(rmax_prob) and helper.is_file_exists(rmax_count):
            self.actions_probs = helper.load_dict_from_file(rmax_prob)
            self.actions_count = helper.load_dict_from_file(rmax_count)
        else:
            self.init_RMax_dict()

        if helper.is_file_exists(q_learning):
            self.data = helper.load_dict_from_file(q_learning)
        else:
            self.create_new_state_action_data("debug_state")
            self.data['debug_state']['actions'] = {"some_action": 45}

    def make_actions_deterministic(self):
        for action_name, action in self.services.parser.actions.items():
            self.deterministic_act_dict[action_name] = {}
            if isinstance(action, parsr_i.ProbabilisticAction):
                for idx in range(len(action.addlists)):
                    if len(action.addlists[idx]) != 0:
                        new_action = parsr_i.Action(idx, action.signature, action.addlists[idx], action.dellists[idx], action.precondition)
                        self.deterministic_act_dict[action_name][idx] = new_action
                    else:
                        self.deterministic_act_dict[action_name][idx] = 'stay_in_place'
            elif isinstance(action, parsr_i.Action):
                self.deterministic_act_dict[action_name][0] = action

    def create_deterministic_domain(self):
        helper.is_directory_exists(policy_files_folder_path)
        lines = helper.make_deterministic(domain_name)
        deterministic_domain_name = policy_files_folder_path + "/" + domain_name
        with open(deterministic_domain_name, 'w') as opened_file:
            opened_file.writelines(lines)

    def change_last_x_states(self, ):
        global LAST_STATE_HASH, CURRENT_STATE_HASH
        if LAST_STATE_HASH == CURRENT_STATE_HASH: return
        if len(self.last_x_states) < self.distance:
            self.last_x_states.append(CURRENT_STATE_HASH)
        else:
            helper.change_indexes(self.last_x_states, CURRENT_STATE_HASH)

    def feel_data(self, state_hash, valid_actions):
        if valid_actions is None: return
        for action in valid_actions:
            checked_act = action.split()[0].split('(')[1]
            if state_hash not in self.data.keys(): self.create_new_state_action_data(state_hash, checked_act, visited=0)
            elif checked_act not in self.data[state_hash]["actions"].keys():
                self.data[state_hash]["actions"][checked_act] = 0

#############################              Start Flags              ###################################
if input_flag == "-L":
    print LocalSimulator().run(domain_name, problem_name, GeneralLearnerAgent())

# elif input_flag == "-E":
#     print LocalSimulator().run(domain_name, problem_name, RMaxAgent())
