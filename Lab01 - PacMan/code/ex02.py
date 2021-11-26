from pacman.util import PriorityQueue
from ex01 import path_reconstruction


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.

    These are the functions to interact with the Pacman world:

    >>> state = problem.getStartState()
    >>> state
    (5, 5)

    >>> problem.getSuccessors(state)
    [((5, 4), 'South', 1), ((4, 5), 'West', 1)]

    >>> problem.isGoalState(state)
    False
    """

    # *** YOUR CODE HERE *** #
    frontier = PriorityQueue()
    start = problem.getStartState()
    frontier.push(start, 0)
    
    explored = {}
    explored[start] = (None, None)

    past = {}
    past[start] = 0

    while not frontier.isEmpty():
        state = frontier.pop()
        if problem.isGoalState(state): break
        for successor, action, cost in problem.getSuccessors(state):
            new_cost = past[state] + cost
            if successor not in explored or new_cost < past[successor]:
                priority = new_cost
                frontier.update(successor, priority)
                explored[successor] = (state, action)
                past[successor] = new_cost


    return []


if __name__ == '__main__':
    import os
    os.system('python -m pacman -a SearchAgent -s uniform_cost_search -l mediumMaze')
    os.system('python -m pacman -a SearchAgent -s uniform_cost_search -l mediumMaze -c stay_east')
    os.system('python -m pacman -a SearchAgent -s uniform_cost_search -l mediumMaze -c stay_west')
