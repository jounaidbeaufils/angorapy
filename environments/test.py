from common.transformers import StateNormalizationTransformer
from common.wrappers import make_env

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = make_env("OpenAIManipulateApproxDiscrete-v0", transformers=[StateNormalizationTransformer])

    state = env.reset()
    done = False
    for j in range(100):
        o, r, d, i = env.step(env.action_space.sample())
        env.render()

    state = env.reset()
    done = False
    for j in range(100):
        o, r, d, i = env.step(env.action_space.sample())
        env.render()