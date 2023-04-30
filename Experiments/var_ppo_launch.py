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

from angorapy.agent.ppo_agent import VarPPOAgent

from angorapy.models.simple import build_var_ffn_models

from angorapy.agent.gather import VarGatherer 

import config_thing as settings

if __name__ == "__main__":
    if settings.var_agent_id is None:
        # build model
        var_agent = VarPPOAgent(model_builder=build_var_ffn_models, environment=settings.env, 
                            horizon=settings.horizon, 
                            workers=settings.workers, 
                            distribution=settings.distribution)
        var_agent.assign_gatherer(VarGatherer)
        if is_root:
            agent_str = var_agent.agent_id + " var"
            settings.store_id(agent_str)

    else:
        #get the model
        var_agent = VarPPOAgent.from_agent_state(settings.var_agent_id)
        var_agent.assign_gatherer(VarGatherer)

    var_agent.drill(n=settings.n, epochs=settings.epochs, batch_size=settings.batch_size, save_every=settings.save_interval)
