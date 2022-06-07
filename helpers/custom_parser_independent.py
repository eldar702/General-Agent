import abc
import random
import copy
from pddlsim import parser_independent as parser_i
REACH_GOAL = 'reach-goal'
## change apply_action_to_state() so the function will get the desire action as argument

class PDDL(object):
    def __init__(self, domain_path, problem_path, domain_name, problem_name, objects, actions, goals, initial_state,
                 failure_conditions=None, revealable_predicates=None):
        """
        :param domain_path: - path of the domain file used
        :param problem_path: - path of the problem file used
        :param domain_name:
        :param problem_name:
        :param objects: objects in the problem, in the format of a dictionary from name to type
        :param actions: action of type Action
        :param goals: a list of Condition
        :param initial_state: the initial state of the problem
        :param failure_conditions the probability that each action will fail
        """
        self.domain_path = domain_path
        self.domain_name = domain_name
        self.problem_path = problem_path
        self.problem_name = problem_name
        self.objects = objects
        self.actions = actions
        self.goals = goals
        self.initial_state = initial_state
        self.failure_conditions = failure_conditions
        self.revealable_predicates = revealable_predicates

        self.uses_custom_features = (not self.failure_conditions is None or len(
            self.goals) > 1)

    def __init__(self, parser, failure_conditions=None, revealable_predicates=None):
        self.domain_path = parser.domain_path
        self.domain_name = parser.domain_name
        self.problem_path = parser.problem_path
        self.problem_name = parser.problem_name
        self.objects = parser.objects
        self.actions = parser.actions
        self.goals = parser.goals
        self.initial_state = parser.initial_state
        self.failure_conditions = failure_conditions
        self.revealable_predicates = revealable_predicates

        self.uses_custom_features = (not self.failure_conditions is None or len(
            self.goals) > 1)

    def build_first_state(self):
        return self.copy_state(self.initial_state)

    def get_object(self, name):
        """ Get a object tuple for a name """
        if name in self.objects:
            return (name, self.objects[name])

    def test_condition(self, condition, mapping):
        return condition.test(mapping)

    def pd_to_strips_string(self, condition):
        return condition.accept(StripsStringVisitor())

    def predicates_from_state(self, state):
        return [("(%s %s)" % (predicate_name, " ".join(map(str, pred)))) for predicate_name, predicate_set in
                state.iteritems() for pred in predicate_set if predicate_name != '=']

    def generate_problem(self, path, state, new_goal):
        predicates = self.predicates_from_state(state)
        goal = self.pd_to_strips_string(new_goal)
        # goal = self.tuples_to_string(new_goal)
        with open(path, 'w') as f:
            f.write('''
    (define (problem ''' + self.problem_name + ''')
    (:domain  ''' + self.domain_name + ''')
    (:objects
        ''')
            for t in self.objects.keys():
                f.write('\n\t' + t)
            f.write(''')
(:init
''')
            f.write('\t' + '\n\t'.join(predicates))
            f.write('''
            )
    (:goal
        ''' + goal + '''
        )
    )
    ''')

    @staticmethod
    def parse_action(action):
        action_sig = action.strip('()').lower()
        parts = action_sig.split(' ')
        action_name = parts[0]
        param_names = parts[1:]
        return action_name, param_names

    def check_action_failure(self, state, action_name):
        for failure in self.failure_conditions:
            if failure.is_relevant(state, action_name) and random.random() < failure.probablity:
                return True
        return False

    def apply_revealable_predicates(self, state):
        for revealable_predicate in self.revealable_predicates:
            if revealable_predicate.is_relevant(state):
                if (random.random() < revealable_predicate.probability):
                    for (predicate_name, entry) in revealable_predicate.effects:
                        state[predicate_name].add(entry)
            else:
                for (predicate_name, entry) in revealable_predicate.effects:
                    predicate_set = state[predicate_name]
                    if entry in predicate_set:
                        predicate_set.remove(entry)

    ## change so this function will get the desire action as argument
    def apply_action_to_state(self, action_sig, state, action, check_preconditions=True):
        action_name, param_names = self.parse_action(action_sig)
        if action_name.lower() == REACH_GOAL:
            return state

        params = map(self.get_object, param_names)

        param_mapping = action.get_param_mapping(params)
        if "food" in action_name:
            print "hello"
        if check_preconditions:
            for precondition in action.precondition:
                if not precondition.test(param_mapping, state):
                    raise PreconditionFalseError()

        if isinstance(action, parser_i.Action):
            for (predicate_name, entry) in action.to_delete(param_mapping):
                predicate_set = state[predicate_name]
                if entry in predicate_set:
                    predicate_set.remove(entry)

            for (predicate_name, entry) in action.to_add(param_mapping):
                state[predicate_name].add(entry)
        else:
            assert isinstance(action, parser_i.ProbabilisticAction)
            index = action.choose_random_effect()
            for (predicate_name, entry) in action.to_delete(param_mapping, index):
                predicate_set = state[predicate_name]
                if entry in predicate_set:
                    predicate_set.remove(entry)

            for (predicate_name, entry) in action.to_add(param_mapping, index):
                state[predicate_name].add(entry)

    def copy_state(self, state):
        return {name: set(entries) for name, entries in state.items()}

    def get_obscure_copy(self, hide_fails=False, hide_probabilistics=False):
        parser_copy = copy.copy(self)
        if hide_fails:
            parser_copy.failure_conditions = None
        if hide_probabilistics:
            for action_name in self.actions.keys():
                cur_action = self.actions[action_name]
                if isinstance(cur_action, parser_i.ProbabilisticAction):
                    max_prob_index = 0
                    for cur_index, cur_prob in enumerate(cur_action.prob_list):
                        if cur_prob > cur_action.prob_list[max_prob_index]:
                            max_prob_index = cur_index
                    self.actions[action_name] = (Action(cur_action.name, cur_action.signature, cur_action.addlists[max_prob_index],
                                                        cur_action.dellists[max_prob_index], cur_action.precondition))

        return parser_copy


class Action(object):
    def __init__(self, name, signature, addlist, dellist, precondition):
        self.name = name
        self.signature = signature
        self.addlist = addlist
        self.dellist = dellist
        self.precondition = precondition

    def action_string(self, dictionary):
        params = " ".join([dictionary[var[0]] for var in self.signature])
        return "(" + self.name + " " + params + ")"

    @staticmethod
    def get_entry(param_mapping, predicate):
        names = [x for x in predicate]
        entry = tuple([param_mapping[name][0] for name in names])
        return entry

    def entries_from_list(self, preds, param_mapping):
        return [(pred[0], self.get_entry(param_mapping, pred[1])) for pred in preds]

    def to_delete(self, param_mapping):
        return self.entries_from_list(self.dellist, param_mapping)

    def to_add(self, param_mapping):
        return self.entries_from_list(self.addlist, param_mapping)

    def get_param_mapping(self, params):
        param_mapping = dict()
        for (name, param_type), obj in zip(self.signature, params):
            param_mapping[name] = obj
        return param_mapping


class ProbabilisticAction(object):
    def __init__(self, name, signature, addlists, dellists, precondition, prob_list):
        self.name = name
        self.signature = signature
        self.addlists = addlists
        self.dellists = dellists
        self.precondition = precondition
        self.prob_list = prob_list

    def action_string(self, dictionary):
        params = " ".join([dictionary[var[0]] for var in self.signature])
        return "(" + self.name + " " + params + ")"

    @staticmethod
    def get_entry(param_mapping, predicate):
        names = [x for x in predicate]
        entry = tuple([param_mapping[name][0] for name in names])
        return entry

    def entries_from_list(self, preds, param_mapping):
        return [(pred[0], self.get_entry(param_mapping, pred[1])) for pred in preds]

    def to_delete(self, param_mapping, effect_index):
        return self.entries_from_list(self.dellists[effect_index], param_mapping)

    def to_add(self, param_mapping, effect_index):
        return self.entries_from_list(self.addlists[effect_index], param_mapping)

    def get_param_mapping(self, params):
        param_mapping = dict()
        for (name, param_type), obj in zip(self.signature, params):
            param_mapping[name] = obj
        return param_mapping

    def choose_random_effect(self):
        """
        Randomly choose effect index according to the prob_list distribution.
        """
        rand = random.uniform(0, 1)
        cur_probs_sum = 0
        for index, prob in enumerate(self.prob_list):
            cur_probs_sum += prob
            if cur_probs_sum > rand:
                print("Effect number {} was randomly selected for action {}".format(index, self.name))
                return index


class Predicate(object):
    def __init__(self, name, signature, negated=False):
        self.name = name
        self.signature = signature
        self.negated = negated

    def ground(self, dictionary):
        return tuple([dictionary[x][0] for x in self.signature])

    def test(self, param_mapping, state):
        result = self.ground(param_mapping) in state[self.name]
        return result if not self.negated else not result


class ConditionVisitor():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def visit_literal(self, literal):
        pass

    @abc.abstractmethod
    def visit_not(self, negation):
        pass

    @abc.abstractmethod
    def visit_conjunction(self, conjunction):
        pass

    @abc.abstractmethod
    def visit_disjunction(self, disjunction):
        pass


class Condition():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def accept(self, visitor):
        pass

    @abc.abstractmethod
    def test(self, mapping):
        pass


class Literal(Condition):
    def __init__(self, predicate, args):
        self.predicate = predicate
        self.args = tuple(args)

    def accept(self, visitor):
        return visitor.visit_literal(self)

    def test(self, mapping):
        return self.args in mapping[self.predicate]


class Truth(Literal):
    def __init__(self):
        self.predicate = 'true'
        self.args = ()

    def accept(self, visitor):
        return visitor.visit_literal(self)

    def test(self, mapping):
        return True


class Falsity(Literal):
    def __init__(self):
        self.predicate = 'false'
        self.args = ()

    def accept(self, visitor):
        return visitor.visit_literal(self)

    def test(self, mapping):
        return False


class Not(Condition):
    def __init__(self, content):
        self.content = content

    def accept(self, visitor):
        return visitor.visit_not(self)

    def test(self, mapping):
        return not self.content.test(mapping)


class JunctorCondition(Condition):
    def __init__(self, parts):
        self.parts = parts


class Conjunction(JunctorCondition):
    def accept(self, visitor):
        return visitor.visit_conjunction(self)

    def test(self, mapping):
        return all([part.test(mapping) for part in self.parts])


class Disjunction(JunctorCondition):
    def accept(self, visitor):
        return visitor.visit_disjunction(self)

    def test(self, mapping):
        return any([part.test(mapping) for part in self.parts])


class StripsStringVisitor(ConditionVisitor):
    def visit_literal(self, condition):
        return "({} {})".format(condition.predicate, ' '.join(condition.args))

    def visit_not(self, condition):
        return "(not {})".format(condition.content.accept(self))

    def join_parts(self, condition):
        return ' '.join([part.accept(self) for part in condition.parts])

    def visit_conjunction(self, condition):
        return "(and {})".format(self.join_parts(condition))

    def visit_disjunction(self, condition):
        return "(or {})".format(self.join_parts(condition))


class FailureCondition():
    def __init__(self, condition, action_names, probablity):
        self.condition = condition
        self.action_names = action_names
        self.probablity = probablity

    def is_relevant(self, state, action_name):
        return action_name in self.action_names and self.condition.test(state)

class RevealablePredicate():
    def __init__(self, condition, effects, probability):
        self.condition = condition
        self.effects = [effect.literal.key for effect in effects]
        self.probability = probability

    def is_relevant(self, state):
        return self.condition.test(state)

class PreconditionFalseError(Exception):
    pass
