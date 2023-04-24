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

# For most environments, PPO needs to normalize states and rewards; to add this functionality we wrap the environment
# with transformers fulfilling this task. You can also add your own custom transformers this way.
wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
env = make_env("LunarLanderContinuous-v2", reward_config=None, transformers=wrappers)

# make policy distribution
distribution = BetaPolicyDistribution(env)

horizon=1024
workers=1

# given the model builder and the environment we can create an agent
var_agent = VarPPOAgent(build_var_ffn_models, env, horizon=horizon, workers=workers, distribution=distribution)

# let's check the agents ID, so we can find its saved states after training
print(f"My Agent's ID: {var_agent.agent_id}")

# set vargatherer
var_agent.assign_gatherer(VarGatherer)

# ... and then train that agent for n cycles
n=10
epochs=3
batch_size=64

var_agent.drill(n=n, epochs=epochs, batch_size=batch_size)

if is_root:
    # after training, we can save the agent for analysis or the like
    var_agent.save_agent_state()

    # render one episode after training
    Investigator.from_agent(var_agent).render_episode(var_agent.env, act_confidently=True)


#repeat original agent 
ori_agent = PPOAgent(build_ffn_models, env, horizon=1024, workers=1, distribution=distribution)
print(f"My Agent's ID: {var_agent.agent_id}")
ori_agent.drill(n=n, epochs=epochs, batch_size=batch_size)
if is_root:
    # after training, we can save the agent for analysis or the like
    ori_agent.save_agent_state()

    # render one episode after training
    Investigator.from_agent(ori_agent).render_episode(ori_agent.env, act_confidently=True)