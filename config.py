import os
from enum import Enum
import argparse

import numpy as np
import random
import torch

from time import gmtime, strftime
import logging
import yaml
import copy
import json
from pprint import pprint

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __new__(cls, args):
        if not isinstance(args, dict):
            return args
        for key in copy.deepcopy(list(args.keys())):
            if "." in key:
                new_keys = key.split(".")
                temp_args = args
                for new_key in new_keys[:-1]:
                    if new_key not in temp_args or not isinstance(temp_args[new_key], dotdict):
                        temp_args[new_key] = dotdict({})
                    temp_args = temp_args[new_key]
                temp_args[new_keys[-1]] = args[key]
            if isinstance(args[key], dict):
                args[key] = dotdict(args[key])
        return super().__new__(cls, args)

def flatten_dict(d, parent_key='', sep='.'):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            yield from flatten_dict(v, new_key, sep=sep)
        else:
            yield new_key, v

class dotformat(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        for key,value in flatten_dict(json.loads(values)):
            setattr(namespace, f"{self.dest}.{key}",value)

def set_config(parser=argparse.ArgumentParser(add_help=False)):

    args = get_args(parser)
    config = dotdict(vars(args))
    seed_everything(config.SEED)
    logging.basicConfig(level=logging.INFO, filename=config.directory.LOGGER_PATH,
                        filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    return config


def get_args(parser):


    #if a perticular arguments present in parser dont add them again
    if not any([arg.dest == 'config_file' for arg in parser._actions]):
        parser.add_argument('-cf','--config_file', default="config.yaml",
                            type=str, help='yaml config file, default: config.yaml')
    
    parser.add_argument('-d','--device', default='cuda:1', type=str,
                        help='cuda:number or cpu, default: cuda:1')
    parser.add_argument('-exp','--experiment', default="unl",
                        type=str, help='name of the experiment default: unl')
    parser.add_argument('-am','--attack_model', default=None, action=dotformat,
                        help='attack model, default: None')
    parser.add_argument('-tr','--trainer', default=None, action=dotformat,
                        help='attack model, default: None')
    parser.add_argument('-atr','--attack_trainer', default=None, action=dotformat,
                        help='attack model, default: None')
    parser.add_argument('-ftr','--finetune', default=None, action=dotformat,
                        help='attack model, default: None')
    parser.add_argument('-rc','--remove_class', default=0, type=int,
                        help='attack model, default: None')

    config_file = parser.parse_args().config_file
    assert os.path.isfile(config_file), "config file not found"

    with open(config_file, "r") as stream:
        defaults = dict(**yaml.safe_load(stream))
        parser.set_defaults(**defaults)

    args = parser.parse_args()


    # dynamic computation on args will be done here
    # convert relative to absolute paths

    # set working dir
    current_time = strftime("%m-%d_0", gmtime())
    args.output_dir = os.path.join(
        'experiments', args.experiment + "_" + current_time)
    if os.path.isdir(args.output_dir):
        while True:
            cur_exp_number = int(args.output_dir[-2:].replace('_', ""))
            args.output_dir = args.output_dir[:-
                                              2] + "_{}".format(cur_exp_number+1)
            if not os.path.isdir(args.output_dir):
                break
    os.makedirs(args.output_dir)
    # once the working dir is set, create the subdirectories
    for key, value in args.directory.items():
        args.directory[key] = os.path.join(args.output_dir, value)

    return args
