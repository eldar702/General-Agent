from copy import deepcopy
import os
import time
import base64
import hashlib
from pddlsim import parser_independent as parsr_i
from helpers import custom_parser_independent as cu_parsr

def init_dict(old_dict, new_val):
    new_dict = deepcopy(old_dict)
    for key, val in old_dict.items():
        for in_key in val.keys():
            new_dict[key][in_key] = new_val
    return new_dict

    #######################            General Helper functions            ############################
def save_dict_to_file(dic, policy_file_name):
    f = open(policy_file_name,'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(policy_file_name):
    f = open(policy_file_name,'r')
    data = f.read()
    f.close()
    return eval(data)

def minute_passed( minutes_number, timer):
    return time.time() - timer >= (60 * minutes_number)


def division_Action(numerator, denominator):
    if denominator == 0:
        return 0
    return float("{:.3f}".format(float(numerator) / float(denominator)))

def is_directory_exists(path):

    if not os.path.exists(path):
        os.makedirs(path)


def make_hash_sha256(o):
    hasher = hashlib.sha256()
    x = make_hashable(o)
    hasher.update(repr(x).encode())
    return base64.b64encode(hasher.digest()).decode()


def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple(sorted(make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o



def make_hash(o):
    if isinstance(o, (set, tuple, list)):
        return tuple([make_hash(e) for e in o])
    elif not isinstance(o, dict):
        return hash(o)
    new_o = deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)
    return hash(tuple(frozenset(sorted(new_o.items()))))

def check_probablistic_instance(instance_1, instance_2):
    return isinstance(instance_1, parsr_i.ProbabilisticAction) and isinstance(instance_2, cu_parsr.ProbabilisticAction)
