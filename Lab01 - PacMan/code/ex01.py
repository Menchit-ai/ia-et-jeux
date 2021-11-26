from pacman.util import Queue


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.

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
    frontier = Queue()
    start = problem.getStartState()
    frontier.push(start)
    explored = {}
    explored[start] = (None,None)

    while not frontier.isEmpty():
        state = frontier.pop()
        if problem.isGoalState(state): break
        for successor, action, cost in problem.getSuccessors(state):
            if not successor in explored:
                frontier.push(successor)
                explored[successor] = (state, action)
    return []


def path_reconstruction(start, goal, explored):
    # *** YOUR CODE HERE *** #
    path = []
    current_state = goal
    while not (current_state == start):
        predecessor, action = explored[current_state]
        path.append(action)
        current_state = predecessor
    path.reverse()
    return path


if __name__ == '__main__':
    import os
    os.system('python -m pacman -a SearchAgent -s breadth_first_search -l tinyMaze')
    os.system('python -m pacman -a SearchAgent -s breadth_first_search -l mediumMaze')
    os.system('python -m pacman -a SearchAgent -s breadth_first_search -l bigMaze -z 0.5')
