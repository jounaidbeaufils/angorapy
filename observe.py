#!/usr/bin/env python
"""Evaluate a loaded agent on a task."""
import argparse
import os
import time

import gym

from agent.ppo_agent import PPOAgent
from analysis.investigation import Investigator
from common.const import BASE_SAVE_PATH, PATH_TO_EXPERIMENTS
from common.wrappers import make_env
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=None)
parser.add_argument("--env", type=str, nargs="?", help="force testing environment", default="")
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="b")
parser.add_argument("--force-case-circulation", action="store_true", help="circle through goal definitions")
parser.add_argument("--rcon", type=str, help="reward configuration", default=None)

args = parser.parse_args()

scale_the_substeps = False

if args.state not in ["b", "best"]:
    args.state = int(args.state)

if args.id is None:
    ids = map(int, [d for d in os.listdir(BASE_SAVE_PATH) if os.path.isdir(os.path.join(BASE_SAVE_PATH, d))])
    args.id = max(ids)

start = time.time()
agent = PPOAgent.from_agent_state(args.id, args.state, force_env_name=None if not args.env else args.env)
print(f"Agent {args.id} successfully loaded.")

try:
    tf.keras.utils.plot_model(agent.joint, to_file=f"{PATH_TO_EXPERIMENTS}/{args.id}/model.png", expand_nested=True,
                              show_shapes=True, dpi=300)
except:
    print("Could not create model plot.")

investigator = Investigator.from_agent(agent)
env = agent.env
if args.env != "":
    env = make_env(args.env, args.rcon)
elif scale_the_substeps:
    parts = env.env.unwrapped.spec.id.split("-")
    new_name = parts[0] + "Fine" + "-" + parts[1]
    print(new_name)
    env = make_env(new_name, args.rcon)

print(f"Evaluating on {env.unwrapped.spec.id} with {env.unwrapped.sim.nsubsteps} substeps.")

if not args.force_case_circulation or (env.unwrapped.spec.id != "FreeReachRelative-v0"):
    for i in range(100):
        investigator.render_episode(env, slow_down=False, substeps_per_step=20 if scale_the_substeps else 1)
else:
    env = gym.make("FreeReachFFRelative-v0")
    for i in range(100):
        env.forced_finger = i % 4
        env.env.forced_finger = i % 4
        env.unwrapped.forced_finger = i % 4
        investigator.render_episode(env, slow_down=False)