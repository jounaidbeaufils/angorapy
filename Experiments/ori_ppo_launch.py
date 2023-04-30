"""
script to run VarPPO from config_thingy.py
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from mpi4py import MPI
    is_root = MPI.COMM_WORLD.rank == 0
except:
    is_root = True

from angorapy.agent.ppo_agent import PPOAgent

from angorapy.models import build_ffn_models

import config_thing as settings

if __name__ == "__main__":
    if settings.ori_agent_id is None:
        # build model
        ori_agent = PPOAgent(build_ffn_models, settings.env, 
                            horizon=settings.horizon, 
                            workers=settings.workers, 
                            distribution=settings.distribution)
        if is_root:
            agent_str = ori_agent.agent_id + " ori"
            settings.store_id(agent_str)

    else:
        #get the model
        ori_agent = PPOAgent.from_agent_state(settings.ori_agent_id)

    ori_agent.drill(n=settings.n, epochs=settings.epochs, batch_size=settings.batch_size, save_every=settings.save_interval)