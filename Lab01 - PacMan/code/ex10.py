import random
from collections import defaultdict
from pacman.learning import ReinforcementAgent
from pacman.util import raiseNotDefined, flipCoin


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_table = defaultdict(lambda: defaultdict(float))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_table[state][action]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        max = 0.0
        for action in self.getLegalActions(state):
          if self.getQValue(state, action) > max : max = self.q_table[state][action]
        return max

    def computeActionFromQValues(self, state):
        """
          Returns the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if "exit" in self.getLegalActions(state) or len(self.getLegalActions(state)) == 0 : return "exit"
        best_actions = []
        for action in self.getLegalActions(state):
          if self.getQValue(state, action) == self.computeValueFromQValues(state):
            best_actions.append(action)

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Returns the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        "*** YOUR CODE HERE ***"
        if "exit" in self.getLegalActions(state) or len(self.getLegalActions(state)) == 0 : return "exit"
        if flipCoin(self.epsilon) : return random.choice(self.getLegalActions(state))
        return self.computeActionFromQValues(state)

    def update(self, state, action, next_state, reward):
        """
          Performs the update of a Q-value with learning rate "self.alpha" and discount rate "self.discount".

          NOTE: You should never call this function. It will be called on your behalf
          when a sample (state, action, next_state, reward) has been collected.
        """
        "*** YOUR CODE HERE ***"
        self.q_table[state][action] = (1 - self.alpha) * self.getQValue(state, action)\
          + self.alpha * (reward + self.discount * self.computeValueFromQValues(next_state))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
