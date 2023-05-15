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
from angorapy.agent.gather import VarGatherer, VarGathererAbs, VarGathererNoPreds
from angorapy.models import build_var_ffn_models

import statistics
import panda_gym
import numpy as np
import itertools

import argparse
import ast

LOG_FILE_PATH = "experiments_log.txt"

parser = argparse.ArgumentParser()

## required arguements ##
parser.add_argument("perm_list",type=int, nargs='+')
args = parser.parse_args()


def permutation():
    #generate permutations of hyperparameters
    c_entropy = [0.1, 0]
    c_var = np.linspace(1.03, 0.9, num=5)
    v_discount = np.linspace(0.1, 0.02, num=5)

    # return the Cartesian product (every combination) of the three sets
    return list(itertools.product(c_entropy, c_var, v_discount))

def train_agent(i: int, permutations):
    # transformers
    wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
    # environment
    env = make_env('PandaReachDense-v2', reward_config=None, transformers=wrappers)

    # policy distribution
    distribution = BetaPolicyDistribution(env)

    agent = VarPPOAgent(model_builder=build_var_ffn_models, environment=env, 
                            horizon=512,
                            workers=1,
                            c_entropy=permutations[i][0],
                            c_var=permutations[i][1],
                            var_discount=permutations[i][2], 
                            distribution=distribution,
                            var_by_adv=True,
                            var_pred=True)
    agent.assign_gatherer(VarGatherer)

    # Run a training loop
    agent.drill(n=300, epochs=3, batch_size=64, save_every=5)
    return agent

def evaluate_agent(agent):
    #  evaluate the performance of the agent
    stats, _ = agent.evaluate(n=10, act_confidently=False, save=True)

    avg_reward = round(statistics.mean(stats.episode_rewards), 2)
    return avg_reward


if __name__ == "__main__":
    for perm_index in args.perm_list:
        for run_num in range(10):
            perm = permutation()
            agent = train_agent(perm_index, perm)
            avg_reward = evaluate_agent(agent)

            # if is_root:
            #     with open(LOG_FILE_PATH, "a") as f:
            #         # Write a new line of text to the file
            #         write_time = time.strftime("%Y-%m-%d %H:%M:%S")
            #         f.write(f"\n{agent.agent_id}, " \
            #                 f"hyperparameter perm: {perm_index}-{run_num}, " \
            #                 f"reward: {avg_reward} ({write_time})")