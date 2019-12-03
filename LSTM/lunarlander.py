import gym
import numpy as np
import random
from keras import utils
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D
from keras.optimizers import Adam
from LSTM.buildingblocks import _buildingblock_network
from models.convolutional import _build_visual_encoder
import tensorflow as tfl

ENV_NAME = "Breakout-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.batch_size = BATCH_SIZE

        #self.model = Sequential()
        #self.model.add(Conv2D(24, 3))
        #self.model.add(MaxPool2D())
        #self.model.add(Dense(16, activation="relu"))
        #self.model.add(Dense(self.action_space, activation="linear"))
        #self.model = _buildingblock_network(self.observation_space, self.action_space,
        #                                    24, self.batch_size)
        self.model = _build_visual_encoder(observation_space)
        self.model.compile(loss="mse",
                           optimizer="Adam")
        #utils.plot_model(self.model, 'breakout_network.png', show_shapes=True)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = np.array(random.sample(self.memory, BATCH_SIZE))
        batch = np.array(self.memory)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                   q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)



env = gym.make(ENV_NAME)

#env = gym.make("CartPole-v1")
#score_logger = ScoreLogger("CartPole-v1")
observation_space = env.observation_space.shape
action_space = env.action_space.n
dqn_solver = DQNSolver(observation_space, action_space)
run = 0
while True:
    run += 1
    state = env.reset()
    state = tfl.expand_dims(state, 0)
    state = tfl.dtypes.cast(state, tfl.float32)
    #state = np.reshape(state, [1, observation_space])
    step = 0
    while True:
        step += 1
        env.render()
        #state.reshape(1, 210, 160, 3)
        action = dqn_solver.act(state)
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        #state_next.reshape(1, 210, 160, 3)
        state_next = tfl.expand_dims(state_next, 0)
        state_next = tfl.dtypes.cast(state_next, tfl.float32)
        #state_next = np.reshape(state_next, [1, observation_space])
        dqn_solver.remember(state, action, reward, state_next, terminal)
        state = state_next
        if terminal:
            print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
            #score_logger.add_score(step, run)
            break
    dqn_solver.experience_replay()
env.close()