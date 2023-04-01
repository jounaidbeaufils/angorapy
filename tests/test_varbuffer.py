from angorapy.common.data_buffers import VarExperienceBuffer
from angorapy.utilities.util import env_extract_dims
from angorapy.common.senses import Sensation

import gym
import numpy as np

env = gym.make("CartPole-v1")

state_dim, action_dim = env_extract_dims(env)

buffer = VarExperienceBuffer(5, state_dim, action_dim, True)

fake_data = np.array([1,2,3,4,5])
fake_sense = [Sensation(proprioception= fake_data) for _ in range(5)]

buffer.fill(fake_sense, fake_data, fake_data, fake_data, fake_data, fake_data, fake_data, fake_data, fake_data)