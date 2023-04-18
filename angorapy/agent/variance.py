import numpy as np
from angorapy.common.const import NP_FLOAT_PREC
from scipy.signal import lfilter

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

# untested function, currently unused. 
def estimate_episode_variance(rewards, values, discount, lam):
    """estimate episode variance in the style of estimate_episode_advantages()"""
    #gets pooled variance. equivalent to deltas
    arrays = np.array(rewards, values[1:], values[:-1])

    # Calculate n_total for each column
    n_total = np.sum(arrays[:, :, 1], axis=0)

    # Calculate variances for each column
    variances = arrays[:, :, 0] * (arrays[:, :, 1] - 1)

    # Sum variances for each column
    sum_variances = np.sum(variances, axis=0)

    # Calculate pooled variances
    pooled_var = sum_variances / (n_total - len(arrays))

    # Combine pooled_var and n_total into a single array
    discounted_var = lfilter([1], [1, float(-(discount * lam))], pooled_var[::-1], axis=0)[::-1].astype(NP_FLOAT_PREC)
    
    return np.vstack((discounted_var, n_total))