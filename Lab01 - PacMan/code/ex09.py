from itertools import cycle
from pacman.util import Counter, raiseNotDefined

def mdp_parameters(type):
    """
    Return the parameters for the DiscountGrid MDP leading to different types of policy:
     a) Prefer the close exit (+1), risking the cliff (-10).
     b) Prefer the close exit (+1), but avoiding the cliff (-10).
     c)	Prefer the distant exit (+10), risking the cliff (-10).
     d)	Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    if type == 'a':
        discount = 0.1
        noise = 0
        reward = 0.1

    elif type == 'b':
        discount = 0.2
        noise = 0.2
        reward = 0.5

    elif type == 'c':
        discount = 0.1
        noise = 0
        reward = 1

    elif type == 'd':
        discount = 0.1
        noise = 0.1
        reward = 1

    return discount, noise, reward

