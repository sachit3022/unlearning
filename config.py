
import os
import torch
from enum import Enum
import numpy as np
import random
import torch


class Config(Enum):
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +"/unlearning"
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    SEED = 2023
    MODEL_DIR = os.path.join(BASE_DIR,"models")
    DATA_PATH = os.path.join(BASE_DIR,"data")
    RESULTS_PATH = os.path.join(BASE_DIR,"results")
    LOG_PATH = os.path.join(BASE_DIR,"logs")

    
random.seed(Config.SEED.value)
np.random.seed(Config.SEED.value)
torch.manual_seed( Config.SEED.value)
torch.cuda.manual_seed_all( Config.SEED.value)