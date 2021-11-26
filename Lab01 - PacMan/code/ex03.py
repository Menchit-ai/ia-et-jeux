from pacman.util import PriorityQueue
from ex01 import path_reconstruction


def astar_search(problem, heuristic=lambda s, p: 0):
    """
    Search the node that has the lowest total cost (past + future) first.

    The heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem. It takes two inputs: a state and a problem.

    >>> heuristic(state, problem)
    11.2

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
                priority = new_cost + heuristic(successor, problem)
                frontier.update(successor, priority)
                explored[successor] = (state, action)
                past[successor] = new_cost

    return path_reconstruction(start, state, explored)


if __name__ == '__main__':
    import os
    os.system('python -m pacman -a SearchAgent -s astar_search -l mediumMaze')
    os.system('python -m pacman -a SearchAgent -s astar_search -f manhattan_heuristic -l mediumMaze')
    os.system('python -m pacman -a SearchAgent -s astar_search -f euclidean_heuristic -c stay_east -l openMaze')
