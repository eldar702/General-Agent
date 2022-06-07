
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
COUNTER = 0
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

    def initialize(self, services):
        self.services = services
        self.create_new_state_action_data("debug_state")
        self.data['debug_state']['actions'] = {"some_action": 45}
        self.make_actions_deterministic()
        self.init_RMax_dict()
        self.new_pddl = cu_parsr.PDDL(self.services.parser)

        pass

    ##########################             General Functions                #################################
    ##########################              Run Agent Run!                  #################################
    def next_action(self):
        global LAST_ACTION, LAST_STATE, CURRENT_STATE, FULL_LAST_ACTION, COUNTER
        CURRENT_STATE = self.services.perception.get_state()
        last_state_hash, current_state_hash = helper.make_hash_sha256(LAST_STATE), helper.make_hash_sha256(CURRENT_STATE)
        chosen_action = None
        self.update_Q_table(last_state_hash)
        self.update_RMax_dict()
   #     self.update_RMax_file()
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
        elif r >= self.epsilon:     # exploit
            chosen_action = self.choose_best_action(valid_actions, current_state_hash)

        LAST_ACTION = chosen_action.split()[0].split('(')[1]
        LAST_STATE = self.services.perception.get_state()
        FULL_LAST_ACTION = chosen_action
        COUNTER += 1
        return chosen_action

    def write_policies_files(self):
        global policy_files_path
        helper.is_directory_exists(policy_files_folder_path)
        helper.save_dict_to_file(self.data, policy_files_path + "-Q-learning")  # save the Q-learning table as file

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
    def create_new_state_action_data(self, key, action=None):
        if action is not None:
            self.data[key] = {'actions': {action: 0}, 'q-val': 0, 'visited': 1 }
        else: self.data[key] = {'actions': {}, 'q-val': 0, 'visited': 1 }

    def feel_data(self, state_hash, valid_actions):
        if valid_actions is None: return
        for action in valid_actions:
            checked_act = action.split()[0].split('(')[1]
            if state_hash not in self.data.keys(): self.create_new_state_action_data(state_hash, checked_act)
            elif checked_act not in self.data[state_hash]["actions"].keys():
                self.data[state_hash]["actions"][checked_act] = 0


    def update_Q_table(self, hash_state):
        global LAST_ACTION, LAST_STATE

        if LAST_ACTION is None:
            return
        reward = self.get_reward()
        if hash_state not in self.data.keys():
            self.create_new_state_action_data(hash_state, LAST_ACTION)
        else:
            self.data[hash_state]["actions"][LAST_ACTION] = ((1 - self.alpha) * self.data[hash_state]["actions"][LAST_ACTION]) + (
                self.alpha * (reward + self.gamma * np.max(self.data[hash_state]["actions"].values())))
        pass

    def change_epsilon(self):
        if self.epsilon > 0.30:
            self.epsilon *= 0.95

    def get_reward(self):
            return 100


    #######################             R-Max  TABLE  Methods               ################################
    def init_RMax_dict(self):
        self.actions_count = helper.init_dict(self.deterministic_act_dict, 0)
        self.actions_probs = copy.deepcopy(self.actions_count)
        pass

    def update_RMax_dict(self):
        global LAST_ACTION
        if LAST_ACTION is None: return

        idx_of_happened_act = self.check_which_act_happened()
        if idx_of_happened_act is not None:
      #      self.actions_count_dict[LAST_ACTION] += 1
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
