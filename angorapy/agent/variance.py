import numpy as np

"""Variance methods providing functionality to the agent."""


def absolute(rewards):
    abs_values = (abs(x) for x in rewards)
    return list(abs_values)

def future_reward_variance(rewards):
# returns the variance of the remaining rewards for each index
    n = len(rewards)
    variance = [0] * n  # Initialize list variance with zeros
    
    #variance
    for i in range(n - 1): #exclude the last index, that is always zero
        variance[i] = np.var(rewards[i:])

    #number of terms
    n_list = [x for x in range(n, -1, -1)]

    return list(zip(variance,n_list))