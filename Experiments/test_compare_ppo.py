"""Script to run PPO and VarPPO, store data and compare"""
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

# For most environments, PPO needs to normalize states and rewards; to add this functionality we wrap the environment
# with transformers fulfilling this task. You can also add your own custom transformers this way.
wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
env = make_env('PandaReachDense-v2', reward_config=None, transformers=wrappers)

# make policy distribution
distribution = BetaPolicyDistribution(env)

# set common variables
horizon=1024
workers=1
n=5
epochs=3
batch_size=64

# build VarPPO
var_agent = VarPPOAgent(build_var_ffn_models, env, horizon=horizon, workers=workers, distribution=distribution)
print(f"My Agent's ID: {var_agent.agent_id}")
var_agent.assign_gatherer(VarGatherer)

# train VarPPO
var_agent.drill(n=n, epochs=epochs, batch_size=batch_size)



#build PPO 
ori_agent = PPOAgent(build_ffn_models, env, horizon=1024, workers=1, distribution=distribution)
print(f"My Agent's ID: {var_agent.agent_id}")

#train PPO
ori_agent.drill(n=n, epochs=epochs, batch_size=batch_size)

if is_root:
    # save agents
    var_agent.save_agent_state()
    ori_agent.save_agent_state()

print("reached script end")