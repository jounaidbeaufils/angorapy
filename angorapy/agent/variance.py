"""Variance methods providing functionality to the agent."""
def absolute(rewards):
    abs_values = (abs(x) for x in rewards)
    return list(abs_values)