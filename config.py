import os
from enum import Enum
import argparse

import numpy as np
import random
import torch

from time import gmtime, strftime
import logging
import yaml


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __new__(cls, args):
        if not isinstance(args, dict):
            return args
        for key, value in args.items():
            if isinstance(value, dict):
                args[key] = dotdict(value)
        return super().__new__(cls, args)


def set_config():

    args = get_args()
    config = dotdict(vars(args))
    seed_everything(config.SEED)
    logging.basicConfig(level=logging.INFO, filename=config.directory.LOGGER_PATH,
                        filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logging.info("config: {}".format(config))
    return config


def get_args():

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config_file', default="config.yaml",
                        type=str, help='yaml config file, default: config.yaml')
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='cuda:number or cpu, default: cuda:1')
    parser.add_argument('--experiment', default="unl",
                        type=str, help='name of the experiment default: unl')

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
            cur_exp_number = int(args.output_dir[-3:].replace('_', ""))
            args.output_dir = args.output_dir[:-
                                              2] + "_{}".format(cur_exp_number+1)
            if not os.path.isdir(args.output_dir):
                break
    os.makedirs(args.output_dir)
    # once the working dir is set, create the subdirectories
    for key, value in args.directory.items():
        args.directory[key] = os.path.join(args.output_dir, value)

    return args
