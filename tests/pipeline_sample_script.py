import gym
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# BUILD DUMMY MODEL
# inputs
proprio_in = tf.keras.Input(batch_shape=(1, 1, 48,), name="proprioception")
touch_in = tf.keras.Input(batch_shape=(1, 1, 92,), name="somatosensation")
vision_in = tf.keras.Input(batch_shape=(1, 1, 200, 200, 3,), name="vision")
goal_in = tf.keras.Input(batch_shape=(1, 1, 15,), name="goal")

# dummy CNN
vmodel = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, 6),
    tf.keras.layers.Conv2D(16, 3, 6),
    tf.keras.layers.Flatten(),
])

vx = tf.keras.layers.TimeDistributed(vmodel)(vision_in)

# concatenate and condense
x = tf.keras.layers.Concatenate()([vx, proprio_in, touch_in, goal_in])
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation=None))(x)

model = tf.keras.Model(inputs=[proprio_in, touch_in, vision_in, goal_in], outputs=[x])


# SIMULATION DUMMY
class NRPDummy:
    """Dummy class representing the role of the NRP/Gazebo simulation."""

    def apply_action(self, action: np.ndarray):
        """Dummy method that acts in place of applying an action in the simulation."""
        pass

    def get_state(self):
        """Dummy method that acts in place of reading the actual data from the NRP."""
        return {"vision": np.random.normal(size=(200, 200, 3)),
                "somatosensation": np.random.normal(size=(92,)),  # touch sensor readings
                "proprioception": np.random.normal(size=(48,)),  # joint positions and velocities
                }

    def reset(self):
        """Dummy method that acts in playe of a full reset of the simulation to initial conditions."""
        pass


# DUMMY ENVIRONMENT
class NRPEnv(gym.Env):
    """Dummy for the environment wrapper around the NRP simulation that we would preferrably construct based on
    openai gym standards"""

    def __init__(self):
        self.sim = NRPDummy()
        self.goal = np.random.random((15,))

        self.action_space = gym.spaces.Box(-1, 1, shape=(20,), dtype=np.float32)

    def _get_obs(self):
        sim_state = self.sim.get_state()

        observation = sim_state.copy()
        observation["goal"] = self.goal

        return observation

    def step(self, action: np.ndarray):
        print(action.shape)
        assert len(action) == 20, "Actions must be 20-dimensional vectors."

        # restrict action based on allowed range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # here, the simulation needs to make a step by following the given action till the next timestep
        self.sim.apply_action(action)

        # after applying the action in simulation, we can read the current state, calculate rewards, etc.
        observation = self._get_obs()
        reward = np.random.normal(0, 1)
        done = False
        info = None

        return observation, reward, done, info

    def reset(self):
        """Reset simulation and get initial state/observation."""
        self.sim.reset()
        return self._get_obs()

    def render(self, mode='human'):
        pass


# HELPER METHODS
def inject_leading_dims(state):
    """Expand state (inplace) to have a batch and time dimension."""
    for sense, value in state.items():
        if value is None:
            continue

        state[sense] = np.expand_dims(value, axis=(0, 1))


# THE LOOP
env = NRPEnv()
state = env.reset()
done = False
while not done:
    inject_leading_dims(state)
    next_action = np.squeeze(model(state))
    state, _, done, _ = env.step(next_action)

    print(state)
