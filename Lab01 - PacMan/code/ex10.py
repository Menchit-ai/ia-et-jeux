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
        self.qvalues = defaultdict(lambda: defaultdict(float))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qvalues[state][action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        qvalue = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if not len(qvalue): return 0.0
        return max(qvalue)

    def computeActionFromQValues(self, state):
        """
          Returns the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        value = self.getValue(state)
        actions = [action for action in self.getLegalActions(state) if self.getQValue(state, action) == value]
        if not len(actions): return None
        return random.choice(actions)

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
        action = None
        
        if flipCoin(self.epsilon):
          action = random.choice(self.getLegalActions(state))
        else:
          action = self.computeActionFromQValues(state)
        
        return action

    def update(self, state, action, next_state, reward):
        """
          Performs the update of a Q-value with learning rate "self.alpha" and discount rate "self.discount".

          NOTE: You should never call this function. It will be called on your behalf
          when a sample (state, action, next_state, reward) has been collected.
        """
        "*** YOUR CODE HERE ***"
        new_qvalue = (1-self.alpha)*self.getQValue(state,action)+self.alpha*(reward+self.discount*self.getValue(next_state))
        self.qvalues[state][action] = new_qvalue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


