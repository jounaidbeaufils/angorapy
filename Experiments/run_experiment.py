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

from angorapy.agent.ppo_agent import PPOAgent,VarPPOAgent
from angorapy.agent.gather import VarGatherer, VarGathererAbs
from angorapy.models import build_var_ffn_models, build_ffn_models

import panda_gym
import argparse
from enum import Enum

LOG_FILE_PATH = "experiments_log.txt"

### Arg Pass ###
parser = argparse.ArgumentParser()

## required arguements ##
parser.add_argument("--exp_str", type=str, required=True)
parser.add_argument("--var_agent", type=bool, required=True) # set false to run the PPOAgent

## arguements with defaults ##
parser.add_argument("--env", type=str, default='PandaReachDense-v2')
parser.add_argument("--horizon", type=int, default=1024)
parser.add_argument("--workers", type=int, default=3)
parser.add_argument("--n", type=int, default=20)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--c_entropy", type=float, default=0.01)
parser.add_argument("--save_interval", type=int, default=64)
parser.add_argument("--agent_id", type=int, default=None)# set ID to load a past agent

## arguements only used with VarPPOAgent ##
parser.add_argument("--c_var", type=float, default=0.001)
parser.add_argument("--abs", type=bool, default=False)
parser.add_argument("--div", type=bool, default=False)

args = parser.parse_args()

### build and load agent ###
def load_agent():
    if args.var_agent:
        agent = VarPPOAgent.from_agent_state(args.agent_id)
        agent.assign_gatherer(VarGatherer)
    else:
        agent = PPOAgent.from_agent_state(args.agent_id)

def build_agent():

    if args.var_agent:

        model_builder = build_ffn_models if args.abs else build_var_ffn_models
        agent = VarPPOAgent(model_builder=model_builder, environment=env, 
                            horizon=args.horizon, 
                            workers=args.workers,
                            c_entropy=args.c_entropy,
                            c_var=args.c_var, 
                            distribution=distribution,
                            var_by_adv=args.div,
                            abs=args.abs)
        
        gatherer = VarGathererAbs if args.abs else VarGatherer
        agent.assign_gatherer(gatherer)
        agent_str = f"var {'div' if args.div else ''} {'abs' if args.abs else ''}"

    else:
        agent = PPOAgent(build_ffn_models, args.env, 
                            horizon=args.horizon, 
                            workers=args.workers,
                            c_entropy=args.c_entropy, 
                            distribution=distribution)
        agent_str = f"ori"
    if is_root:
        with open(LOG_FILE_PATH, "a") as f:
            # Write a new line of text to the file
            write_time = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{agent.agent_id} {agent_str} ({args.exp_str} {write_time})")
    
    return agent

if __name__ == "__main__":
    ### Run Script ###
    # transformers
    wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
    # environment
    env = make_env(args.env, reward_config=None, transformers=wrappers)

    # policy distribution
    distribution = BetaPolicyDistribution(env)
    
    # build or load agent
    if args.agent_id is None:
        agent = build_agent()
    else:
        agent = load_agent()

    # train agent
    agent.drill(n=args.n, epochs=args.epochs, batch_size=args.batch_size, save_every=args.save_interval)
    