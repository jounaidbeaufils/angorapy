"""
run any model with argparses
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
from angorapy.agent.gather import VarGatherer, VarGathererAbs, VarGathererNoPreds, VarGathererNoise
from angorapy.models import build_var_ffn_models, build_ffn_models

import panda_gym
import argparse
from enum import Enum

LOG_FILE_PATH = "experiments_log.txt"

### Arg Pass helper ###
def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False

### Arg Pass ###
parser = argparse.ArgumentParser()

## required arguements ##
parser.add_argument("exp_str", type=str)
parser.add_argument("gather_type", type=str, choices=["var_pred", "var_no_pred", "abs", "noise", "ori"])

## arguements with defaults ##
parser.add_argument("--env", type=str, default='PandaPushDense-v2')
parser.add_argument("--horizon", type=int, default=512)
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--n", type=int, default=20)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--c_entropy", type=float, default=0.01)
parser.add_argument("--save_interval", type=int, default=64)
parser.add_argument("--agent_id", type=int, default=None)# set ID to load a past agent

## arguements only used with VarPPOAgent ##
parser.add_argument("--c_var", type=float, default=0.001)
parser.add_argument("--var_discount", type=float, default=0.99)
parser.add_argument("--div", type=str_to_bool, default=False)

args = parser.parse_args()

class GathererEnum(Enum):
        VAR_PRED = VarGatherer
        ABS = VarGathererAbs
        VAR_NO_PRED = VarGathererNoPreds
        NOISE = VarGathererNoise
        CLASSIC = None

### build and load agent ###
def load_agent():
    if args.gather_type == "ori":
        agent = PPOAgent.from_agent_state(args.agent_id)
    else: 
        agent = VarPPOAgent.from_agent_state(args.agent_id)
        gatherer = getattr(GathererEnum, args.gather_type.upper()).value
        agent.assign_gatherer(gatherer)


def build_agent():
    if args.gather_type == "ori":
        agent = PPOAgent(build_ffn_models, environment=env, 
                    horizon=args.horizon, 
                    workers=args.workers,
                    c_entropy=args.c_entropy, 
                    distribution=distribution)
        agent_str = f"ori"
    else:
        var_pred = True if args.gather_type == "var_pred" else False
        model_builder = build_var_ffn_models if var_pred else build_ffn_models
        agent = VarPPOAgent(model_builder=model_builder, environment=env, 
                            horizon=args.horizon, 
                            workers=args.workers,
                            c_entropy=args.c_entropy,
                            c_var=args.c_var,
                            var_discount=args.var_discount, 
                            distribution=distribution,
                            var_by_adv=args.div,
                            var_pred=var_pred)
        
        gatherer = getattr(GathererEnum, args.gather_type.upper()).value
        agent.assign_gatherer(gatherer)
        agent_str = f"{args.gather_type}"

    # if is_root:
    #     with open(LOG_FILE_PATH, "a") as f:
    #         # Write a new line of text to the file
    #         write_time = time.strftime("%Y-%m-%d %H:%M:%S")
    #         f.write(f"\n{agent.agent_id} {agent_str} ({args.exp_str} {write_time})")
    
    return agent


if __name__ == "__main__":
    ### Run Script ###
    # transformers
    wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
    # environment
    env = make_env(args.env, reward_config=None, transformers=wrappers, render=False)

    # policy distribution
    distribution = BetaPolicyDistribution(env)
    
    # build or load agent
    if args.agent_id is None:
        agent = build_agent()
    else:
        agent = load_agent()

    # train agent
    agent.drill(n=args.n, epochs=args.epochs, batch_size=args.batch_size, save_every=args.save_interval)
    