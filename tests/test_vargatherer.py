"""Example script showcasing how to train an agent."""
import os

from angorapy.analysis.investigation import Investigator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from mpi4py import MPI
    is_root = MPI.COMM_WORLD.rank == 0
except:
    is_root = True

from angorapy.agent.ppo_agent import PPOAgent, VarPPOAgent
from angorapy.common.policies import BetaPolicyDistribution
from angorapy.common.transformers import RewardNormalizationTransformer, StateNormalizationTransformer
from angorapy.common.wrappers import make_env

from angorapy.models import get_model_builder, build_ffn_models

from angorapy.agent.gather import VarGatherer

# For most environments, PPO needs to normalize states and rewards; to add this functionality we wrap the environment
# with transformers fulfilling this task. You can also add your own custom transformers this way.
wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
env = make_env("LunarLanderContinuous-v2", reward_config=None, transformers=wrappers)

# make policy distribution
distribution = BetaPolicyDistribution(env)

# the agent needs to create the model itself, so we build a method that builds a model
build_models = build_ffn_models

# given the model builder and the environment we can create an agent
agent = VarPPOAgent(build_models, env, horizon=1024, workers=1, distribution=distribution)

# let's check the agents ID, so we can find its saved states after training
print(f"My Agent's ID: {agent.agent_id}")

# set vargatherer
agent.assign_gatherer(VarGatherer)

# ... and then train that agent for n cycles
agent.drill(n=10, epochs=3, batch_size=64)

if is_root:
    # after training, we can save the agent for analysis or the like
    agent.save_agent_state()

    # render one episode after training
    Investigator.from_agent(agent).render_episode(agent.env, act_confidently=True)
