
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
from helpers import custom_parser_independent as cu_parsr
import helpers.helper_functions as helper


# read files:
input_flag = sys.argv[1]
domain_name = sys.argv[2]
problem_name = sys.argv[3]
policy_files_folder_path = os.getcwd() + ("/" + domain_name).replace(".pddl", "", 2)
policy_files_path = (os.getcwd() + ("/" + domain_name) + "/" + problem_name).replace(".pddl", "", 3)

# GLOBALS
TIMER = time.time()
LAST_STATE = None
LAST_ACTION = None
CURRENT_STATE = None
FULL_LAST_ACTION = None
LAST_STATE_HASH = None
COUNTER = 0
DISTANCE_TO_CHECK = 10
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
        self.distance = DISTANCE_TO_CHECK

    def initialize(self, services):
        self.services = services
        self.make_actions_deterministic()
        self.read_policies_files()

        self.new_pddl = cu_parsr.PDDL(self.services.parser)
        self.max_reward = self.uncompleted_goals = self.get_uncompleted_goals(self.services.goal_tracking.uncompleted_goals[0])

        pass

    ##########################             General Functions                #################################
    ##########################              Run Agent Run!                  #################################
    def next_action(self):
        global LAST_ACTION, LAST_STATE, CURRENT_STATE, FULL_LAST_ACTION, LAST_STATE_HASH, COUNTER
        CURRENT_STATE = self.services.perception.get_state()
        LAST_STATE_HASH, current_state_hash = helper.make_hash_sha256(LAST_STATE), helper.make_hash_sha256(CURRENT_STATE)
        chosen_action = None
        self.update_Q_table(), self.update_RMax_dict()
        self.write_policies_files()

        r = np.random.uniform(0, 1)
        self.change_epsilon()
        if self.services.goal_tracking.reached_all_goals() and helper.minute_passed(0.1, TIMER):
            return None

        valid_actions = self.services.valid_actions.get()
        self.feel_data(current_state_hash, valid_actions)
        if len(valid_actions) == 0: return None
        elif len(valid_actions) == 1:
            chosen_action = self.services.valid_actions.get()[0]
        elif r < self.epsilon:      # explore
            chosen_action = random.choice(valid_actions)
         #   chosen_action = self.choose_curiosity_random(valid_actions)
        elif r >= self.epsilon:     # exploit
            chosen_action = self.choose_best_action(valid_actions, current_state_hash)

        LAST_ACTION = chosen_action.split()[0].split('(')[1]
        LAST_STATE = self.services.perception.get_state()
        FULL_LAST_ACTION = chosen_action
        COUNTER += 1
        return chosen_action
      #  self.services.parser.apply_revealable_predicates(current_state)

    def write_policies_files(self):
        global policy_files_path
        helper.is_directory_exists(policy_files_folder_path)
        helper.save_dict_to_file(self.data, policy_files_path + "-Q-learning")  # save the Q-learning table as file
        helper.save_dict_to_file(self.actions_count, policy_files_path + "-Rmax-count")  # save the actions count
        helper.save_dict_to_file(self.actions_probs, policy_files_path + "-Rmax-prob")  # save the actions probabilities

        ####################             Q - LEARNING  Methods               #############################
    def choose_best_action(self, valid_Actions, state_hash):

        best_action = []
        best_action_value = float('-inf')
        for action in valid_Actions:
            checked_act = action.split()[0].split('(')[1]
            action_value = self.data[state_hash]["actions"][checked_act]  # the checked action is not new for this state
            if action_value == best_action_value:
                best_action.append(action)
            elif action_value > best_action_value:
                best_action = [action]

        return random.choice(best_action)

    #######################             Q-table  Methods               ################################
    def create_new_state_action_data(self, key, action=None, visited=1):
        if action is not None:
            self.data[key] = {'actions': {action: 0}, 'dist_from_goal': self.distance, "is_goal": 0, 'visited': visited }
        else: self.data[key] = {'actions': {}, 'dist_from_goal': self.distance, "is_goal": 0, 'visited': visited }

    def feel_data(self, state_hash, valid_actions):
        if valid_actions is None: return
        for action in valid_actions:
            checked_act = action.split()[0].split('(')[1]
            if state_hash not in self.data.keys(): self.create_new_state_action_data(state_hash, checked_act, visited=0)
            elif checked_act not in self.data[state_hash]["actions"].keys():
                self.data[state_hash]["actions"][checked_act] = 0


    def update_Q_table(self):
        global LAST_ACTION, LAST_STATE
        if LAST_ACTION is None: return
        if LAST_STATE_HASH not in self.data.keys():
            self.create_new_state_action_data(LAST_STATE_HASH, action=LAST_ACTION)

        if len(self.services.goal_tracking.uncompleted_goals) == 0:
            reward = self.max_reward - 1    #################################     ########## NEED TO CHANGE
        else:
            uncompleted_goals_number = self.services.goal_tracking.uncompleted_goals[0]
            reward = self.get_reward(uncompleted_goals_number)
        print(reward)

        self.data[LAST_STATE_HASH]["actions"][LAST_ACTION] = ((1 - self.alpha) * self.data[LAST_STATE_HASH]["actions"][LAST_ACTION]) + (
            self.alpha * (reward + self.gamma * np.max(self.data[LAST_STATE_HASH]["actions"].values())))
        pass

    def change_epsilon(self):
        if self.epsilon > 0.50:
            self.epsilon *= 0.95


    def choose_curiosity_random(self, valid_actions):
        global CURRENT_STATE
        best_action = []
        for action in valid_actions:
            action_name = action.split()[0].split('(')[1]
            for deterministic_act in self.deterministic_act_dict[action_name]:
                temp_state = copy.deepcopy(CURRENT_STATE)
                cu_parsr.PDDL.apply_action_to_state(self.new_pddl, action, temp_state, deterministic_act, check_preconditions=False)
                hash_state = helper.make_hash_sha256(CURRENT_STATE)
                if hash_state not in self.data.keys():
                    best_action.append(action)
        if len(best_action) == 0:
            return random.choice(valid_actions)

    #######################             R-Max  TABLE  Methods               ################################
    def init_RMax_dict(self):
        self.actions_count = helper.init_dict(self.deterministic_act_dict, 0)
        self.actions_probs = copy.deepcopy(self.actions_count)

    def update_RMax_dict(self):
        global LAST_ACTION
        if LAST_ACTION is None: return

        idx_of_happened_act = self.check_which_act_happened()
        if idx_of_happened_act is not None:
            self.actions_count[LAST_ACTION][idx_of_happened_act] += 1
        self.calculate_probabilities()

    def check_which_act_happened(self):
        global CURRENT_STATE, LAST_STATE, LAST_ACTION, FULL_LAST_ACTION
        if LAST_ACTION is None: return
        hash_current_state = helper.make_hash_sha256(CURRENT_STATE)
        for idx, action in self.deterministic_act_dict[LAST_ACTION].items():
            if action == "stay_in_place":
                hash_temp_state = helper.make_hash_sha256(LAST_STATE)
                if hash_current_state == hash_temp_state: return idx
            else:
                temp_state = copy.deepcopy(LAST_STATE)
                cu_parsr.PDDL.apply_action_to_state(self.new_pddl, FULL_LAST_ACTION, temp_state, action, check_preconditions=False)
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
                else: self.actions_probs[action_name][idx] = 0

    ##############################             Helpers  Methods               #################################
    def read_policies_files(self):
        global policy_files_path
        rmax_prob, rmax_count = (policy_files_path + "-Rmax-prob"), (policy_files_path + "-Rmax-count")
        q_learning = policy_files_path + "-Q-learning"

        if helper.is_file_exists(rmax_prob) and helper.is_file_exists(rmax_count):
            self.actions_probs = helper.load_dict_from_file(rmax_prob)
            self.actions_count = helper.load_dict_from_file(rmax_count)
        else: self.init_RMax_dict()

        if helper.is_file_exists(q_learning): self.data = helper.load_dict_from_file(q_learning)
        else:
            self.create_new_state_action_data("debug_state")
            self.data['debug_state']['actions'] = {"some_action": 45}


    ######################             Reward Functions               ##############################

    def get_uncompleted_goals(self, goal, reward=0.0):
        if isinstance(goal, parsr_i.Literal):
            return 1.0
        if isinstance(goal, parsr_i.Conjunction):
            for part_goal in goal.parts:
                reward += self.get_uncompleted_goals(part_goal)
        elif isinstance(goal, parsr_i.Disjunction):
            max_reward = 0.0
            for part_goal in goal.parts:
                temp_reward = self.get_uncompleted_goals(part_goal)
                if temp_reward > max_reward:
                    max_reward = temp_reward
            reward += max_reward
        return reward

    def get_reward(self, goal):
        global LAST_STATE_HASH
        uncompleted_goals_num = self.get_uncompleted_goals(goal)
        self.check_if_found_goal(uncompleted_goals_num)
        if self.data[LAST_STATE_HASH]["dist_from_goal"]:
            return 10
        else:
        return self.max_reward - self.get_uncompleted_goals(goal) - 1

    def check_if_found_goal(self, uncompleted_goals_num):
        global LAST_STATE_HASH
        if uncompleted_goals_num < self.uncompleted_goals:
            self.uncompleted_goals = uncompleted_goals_num
            self.data[LAST_STATE_HASH]["is_goal"] = True
            return True
        return False

    ##############################             PDDL  Methods               #################################

    def is_probabilistic(self, action):
        return isinstance(self.services.valid_actions.provider.parser.actions[action], parsr_i.ProbabilisticAction)

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




#############################              Start Flags              ###################################
if input_flag == "-L":
    print LocalSimulator().run(domain_name, problem_name, GeneralLearnerAgent())

# elif input_flag == "-E":
#     print LocalSimulator().run(domain_name, problem_name, RMaxAgent())
