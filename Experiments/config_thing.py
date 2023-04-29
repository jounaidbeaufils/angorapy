"""
Stores a variables that are shared between ori_ppo_launch.py and var_launch_ppo.py
"""

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from mpi4py import MPI
    is_root = MPI.COMM_WORLD.rank == 0
except:
    is_root = True

from angorapy.common.policies import BetaPolicyDistribution
from angorapy.common.transformers import RewardNormalizationTransformer, StateNormalizationTransformer
from angorapy.common.wrappers import make_env


import panda_gym



LOG_FILE_PATH = "experiments_log.txt"

### Default tranformers, distribution and variables ###
# transformer
wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
# environment
env = make_env('PandaReachDense-v2', reward_config=None, transformers=wrappers)

# policy distribution
distribution = BetaPolicyDistribution(env)

# common variables
horizon=1024
workers=3
n=110
epochs=3
batch_size=64
save_interval = 3

### Experiment tage ###
experiment_str = "simultaneuos mpi test run" # insert experiment tag

### Model details ###
#use to continue training an older model
#set to None to make new agent
ori_agent_id = 1682774068794780  
var_agent_id = 1682781944436042   

def store_id(agent_id_str):
    # Open the file in append mode, creating it if it doesn't exist
    with open(LOG_FILE_PATH, "a") as f:
        # Write a new line of text to the file
        write_time = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n {agent_id_str} ({experiment_str} {write_time})")