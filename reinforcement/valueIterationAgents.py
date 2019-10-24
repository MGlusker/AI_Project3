# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        
        # keep track of new values
        #newValues = util.Counter()

        # need to find new utilites for all states in S
        states = mdp.getStates()

        # update the values for all iterations
        for i in range(iterations):
          
          newValues = util.Counter()
          # for each state find the new utility
          for s in states:

            # if we're at a terminal state
            if mdp.isTerminal(s):
              newValues[s] = self.mdp.getReward(s, None, None)

            else:   
              #find possible actions
              actions = mdp.getPossibleActions(s)

              # as long as there are actions to take
              #if(not mdp.isTerminal(s)):
              if len(actions) != 0:
                
                
                # update utilities by finding the maximum q value
                # largestUtility = -float("Inf")
                # find qValues based on the actions
                qValues = []
                for a in actions:
                  transitionStates = self.mdp.getTransitionStatesAndProbs(s, a)

                  sumOfTransitions = 0.0
                  
                  for ts in transitionStates:
                    nextState = ts[0]
                    prob = ts[1]
                    sumOfTransitions += prob * (self.mdp.getReward(s, a, nextState) + (self.discount * self.getValue(nextState)))

                  qValue = sumOfTransitions

                  qValues.append(qValue)

                  
                  # qValue = self.computeQValueFromValues(s, a)

                  # if qValue > largestUtility:
                  #  largestUtility = qValue

                #newValues[s] = largestUtility
                maxQValue = max(qValues)
                newValues[s] = maxQValue

          self.values = newValues



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
       
        # if your state is terminal state
        #if self.mdp.isTerminal(state):
        #  return None

        # get transition states from state
        transitionStates = self.mdp.getTransitionStatesAndProbs(state, action)

        sumOfTransitions = 0.0

        for ts in transitionStates:
          nextState = ts[0]
          prob = ts[1]
          sumOfTransitions += prob * (self.mdp.getReward(state, action, nextState) + (self.discount * self.getValue(nextState)))

        qValue = sumOfTransitions

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        

        actions = self.mdp.getPossibleActions(state)

        maxQValue = -float("Inf")
        bestAction = None

        for a in actions:
          currentQValue = self.computeQValueFromValues(state, a)

          if currentQValue > maxQValue:
            bestAction = a 
            maxQValue = currentQValue

        return bestAction

        """
        actions = self.mdp.getPossibleActions(state)

        # if we're at a terminal state
        if len(actions) == 0:
          return None

        # find transition states
        #transitionStates = None
        bestAction = None
        maxActionScore = -float("Inf")

        for a in actions:
          transitionStates = self.mdp.getTransitionStatesAndProbs(state, a)

        #bestNextState = None
        #largestUtility = -float("Inf")
        #transitionStateScore = 0.0

          transitionStateScore = 0.0
          # calculate the utility of each of the transition states
          for t in transitionStates:
            
            nextState = t[0]
            prob = t[1]
            transitionStateScore += prob * (self.mdp.getReward(state, a, nextState) + (self.discount * self.getValue(nextState))) 
            #transitionStateScore += prob * self.getValue(nextState) 

            if transitionStateScore > maxActionScore: #largestUtility:
              #largestUtility = transitionStateScore
              #bestNextState = nextState
              maxActionScore = transitionStateScore
              bestAction = a


        return bestAction
        """

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
