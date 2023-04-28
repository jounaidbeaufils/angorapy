"""Script to run PPO and VarPPO, and store the agent"""
import os

from angorapy.analysis.investigation import Investigator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from mpi4py import MPI
    is_root = MPI.COMM_WORLD.rank == 0
except:
    is_root = True

from angorapy.agent.ppo_agent import VarPPOAgent, PPOAgent
from angorapy.common.policies import BetaPolicyDistribution
from angorapy.common.transformers import RewardNormalizationTransformer, StateNormalizationTransformer
from angorapy.common.wrappers import make_env

from angorapy.models import build_ffn_models
from angorapy.models.simple import build_var_ffn_models

from angorapy.agent.gather import VarGatherer

import panda_gym

import time
import random

#used to store IDs in a text file
start_time = time.strftime("%Y%m%d-%H%M%S")
random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
file_name = f"{start_time}_{random_string}"


### Default tranformers, distribution and variables ###
# transformer
wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
# environment
env = make_env('PandaReachDense-v2', reward_config=None, transformers=wrappers)

# policy distribution
distribution = BetaPolicyDistribution(env)

# common variables
horizon=1024
workers=1
n=5
epochs=3
batch_size=64




def build_models():
    ### Used to store IDs in a text file ###
    start_time = time.strftime("%Y%m%d-%H%M%S")
    random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
    file_name = f"{start_time}_{random_string}"
    print(f"model ID saved to {file_name}")

    ### Build Models ###
    # build VarPPO
    var_agent = VarPPOAgent(build_var_ffn_models, env, horizon=horizon, workers=workers, distribution=distribution)
    var_id_str = str(var_agent.agent_id) + " (var)"
    store_id(var_id_str, file_name)
    print(f"Just built agent: {(var_id_str)} \n")
    var_agent.assign_gatherer(VarGatherer)

    #build PPO 
    ori_agent = PPOAgent(build_ffn_models, env, horizon=horizon, workers=workers, distribution=distribution)
    ori_id_str = str(ori_agent.agent_id) + " (ori)"
    store_id(ori_id_str, file_name)
    print(f"Just built agent: {ori_id_str} \n")

    return ori_agent, var_agent

def train_save_models(save_interval, ori_agent: PPOAgent, var_agent: VarPPOAgent):
    ### train and save Model ###
    #train PPO
    print(f"Drilling agent: {ori_agent.agent_id}")
    ori_agent.drill(n=n, epochs=epochs, batch_size=batch_size, save_every=save_interval)

    # train VarPPO
    print(f"Drilling agent: {var_agent.agent_id}")
    var_agent.drill(n=n, epochs=epochs, batch_size=batch_size,save_every=save_interval)

def store_id(agent_id_str, file_name):
    # Open the file in append mode, creating it if it doesn't exist
    with open(f"Experiments/{file_name}.txt", "a") as f:
        # Write a new line of text to the file
        f.write(f"\n {agent_id_str}")

if __name__ == "__main__":
    ori, var = build_models()
    train_save_models(3, ori, var)

    print("Training done")

### testing agent retrival ###
#re_ori = PPOAgent.from_agent_state(ori.agent_id, "b")
#print(f"just loaded {ori.agent_id}")
#stats, _ = re_ori.evaluate(n=10, act_confidently=False)
#print(stats.episode_lengths)

#re_var = VarPPOAgent.from_agent_state(var.agent_id, "b")
#print(f"just loaded {var.agent_id}")
#stats, _ = re_var.evaluate(n=10, act_confidently=False)
#print(stats.episode_lengths)
