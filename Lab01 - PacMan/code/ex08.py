from collections import defaultdict
from pacman.util import raiseNotDefined


def value_iteration(mdp, discount, iterations):
    """
    Value iteration is an algorithm that estimates the Q-values of an MDP.
    It runs for the given number of iterations, using the supplied discount factor.

    Some useful MDP methods you will use:
        mdp.getStates()
        mdp.getPossibleActions(state)
        mdp.getTransitionStatesAndProbs(state, action)
        mdp.getReward(state, action, nextState)
        mdp.isTerminal(state)
    """
    q_table = defaultdict(lambda: defaultdict(float))  # dict of dicts with default 0
    v_table = defaultdict(float)

    for _ in range(iterations):
        for state in mdp.getStates():
            if mdp.isTerminal(state): v_table[state] = 0
            else :
                max = -1
                for action in mdp.getPossibleActions(state):
                    if q_table[state][action] > max : max = q_table[state][action]
                v_table[state] = max

        for state in mdp.getStates():
            for action in mdp.getPossibleActions(state):
                q = 0
                for next_state, prob in mdp.getTransitionStatesAndProbs(state,action):
                    R = mdp.getReward(state, action, next_state)
                    q += prob * (R + discount * v_table[next_state])
                q_table[state][action] = q

    return q_table
