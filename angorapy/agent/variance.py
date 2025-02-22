import numpy as np
from angorapy.common.const import NP_FLOAT_PREC
from scipy.signal import lfilter
import random

"""Variance methods providing functionality to the agent."""

def noise(rewards, values, discount, lam):
    noise_values = (random.random() for _ in range(len(rewards)))
    return list(noise_values)

def absolute(rewards, values, discount, lam):
    abs_values = (abs(x) for x in rewards)
    return list(abs_values)

def future_reward_variance(rewards, values=None, discount=None, lam=None):
# returns the variance of the remaining rewards for each index
    n = len(rewards)
    variance = [0] * n  # Initialize list variance with zeros
    
    #variance
    for i in range(n - 1): #exclude the last index, that is always zero
        variance[i] = np.var(rewards[i:])

    #number of terms
    n_list = [x for x in range(n, -1, -1)]

    if None in [values, discount, lam]:
        return list(zip(variance,n_list))
    else:
        # apply discount, function is being used by gatherer (yes this is bad code, i should uncouple this)
        discounted_var = lfilter([1], [1, float(-(discount * lam))], variance[::-1], axis=0)[::-1].astype(NP_FLOAT_PREC)
        return discounted_var

def estimate_episode_variance(rewards, values, discount, lam):
    """estimate episode variance in the style of estimate_episode_advantages()"""
    # calculate variances of rewards list
    reward_variance = future_reward_variance(rewards)

    # convert to numpy
    array1_np = np.array(reward_variance)
    array2_np = np.array(values[:-1])
    array3_np = np.array(values[1:])

    # spilt into a variance and sample arrays
    variances = np.vstack((array1_np[:, 0], array2_np[:, 0], array3_np[:, 0]))
    sample_sizes = np.vstack((array1_np[:, 1], array2_np[:, 1], array3_np[:, 1]))

    # pooled variance (equivalent to weighteed mean of variances)
    pooled_variances = (np.sum((sample_sizes - 1) * variances, axis=0) / (np.sum(sample_sizes, axis=0) - 3)).reshape(-1, 1)

    # sum samples sizes
    total_sample_sizes = np.sum(sample_sizes, axis=0).reshape(-1, 1)

    # apply discount
    discounted_var = lfilter([1], [1, float(-(discount * lam))], pooled_variances[::-1], axis=0)[::-1].astype(NP_FLOAT_PREC)
    
    # Combine pooled_var and n_total into a single array
    result = np.hstack((discounted_var, total_sample_sizes))

    return [tuple(row) for row in result]