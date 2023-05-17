import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from mpi4py import MPI

    is_root = MPI.COMM_WORLD.rank == 0
except:
    is_root = True

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from angorapy.common.policies import BetaPolicyDistribution
from angorapy.common.transformers import RewardNormalizationTransformer, StateNormalizationTransformer
from angorapy.common.wrappers import make_env

from angorapy.agent.ppo_agent import VarPPOAgent
from angorapy.agent.gather import VarGatherer
from angorapy.models import build_var_ffn_models

import statistics
import numpy as np
import itertools

import argparse

import panda_gym

LOG_FILE_PATH = "experiments_log.txt"

parser = argparse.ArgumentParser()

## required arguements ##
parser.add_argument("--env", type=str, default='PandaPushDense-v2')
parser.add_argument("perm_list", type=int, nargs='+')
args = parser.parse_args()


def permutation():
    # generate permutations of hyperparameters
    c_entropy = [0.1, 0]
    c_var = np.linspace(1.03, 0.9, num=5)
    v_discount = np.linspace(0.1, 0.02, num=5)

    # return the Cartesian product (every combination) of the three sets
    return list(itertools.product(c_entropy, c_var, v_discount))


def train_agent(i: int, permutations):
    # transformers
    wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
    # environment
    env = make_env(args.env, reward_config=None, transformers=wrappers, render=False)

    # policy distribution
    distribution = BetaPolicyDistribution(env)

    agent = VarPPOAgent(model_builder=build_var_ffn_models, environment=env,
                        horizon=512,
                        workers=12,
                        c_entropy=permutations[i][0],
                        c_var=permutations[i][1],
                        var_discount=permutations[i][2],
                        distribution=distribution,
                        var_by_adv=True,
                        var_pred=True)
    agent.assign_gatherer(VarGatherer)

    # Run a training loop
    agent.drill(n=300, epochs=3, batch_size=64)
    return agent


def evaluate_agent(agent):
    #  evaluate the performance of the agent
    stats, _ = agent.evaluate(n=24, act_confidently=False, save=True)

    avg_reward = round(statistics.mean(stats.episode_rewards), 2)

    print(f"Evaluated agent {agent.agent_id} with average reward {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    for perm_index in args.perm_list:
        for run_num in range(10):
            perm = permutation()
            agent = train_agent(perm_index, perm)

            best_agent = VarPPOAgent.from_agent_state(agent.agent_id, from_iteration="best")
            avg_reward = evaluate_agent(best_agent)

            # if is_root:
            #     with open(LOG_FILE_PATH, "a") as f:
            #         # Write a new line of text to the file
            #         write_time = time.strftime("%Y-%m-%d %H:%M:%S")
            #         f.write(f"\n{agent.agent_id}, " \
            #                 f"hyperparameter perm: {perm_index}-{run_num}, " \
            #                 f"reward: {avg_reward} ({write_time})")
