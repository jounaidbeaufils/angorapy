import numpy as np

"""Variance methods providing functionality to the agent."""


def absolute(rewards):
    abs_values = (abs(x) for x in rewards)
    return list(abs_values)

def future_reward_variance(rewards):
    n = len(rewards)
    variance = [0] * n  # Initialize list variance with zeros
    
    for i in range(n):
        if i == n - 1:  # If i is the last index, variance is zero
            variance[i] = 0
        else:
            variance[i] = np.var(rewards[i+1:])
    
    return variance


def pooled_variance(var_list1, var_list2): #not working

    total_n = len(var_list1)
    pooled_var = []

    for i, var1, var2, n2 in enumerate(var_list1, var_list2):
        n1 = total_n - i
        pooled_var[i] = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)

    return pooled_var
