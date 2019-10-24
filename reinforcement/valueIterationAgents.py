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
        self.newValues = util.Counter()

        states = mdp.getStates()

       
        # Write value iteration code here
        for i in range(iterations):
          print "***", i 
          for s in states:

            actions = mdp.getPossibleActions(s)

            if(len(actions) !=0):
              max = -float('Inf')
              for a in actions:

                qValue = self.computeQValueFromValues(s, a)

                if qValue > max: 
                  max = qValue

              self.newValues[s] = max

        self.values = self.newValues


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
        
        transitionStates = self.mdp.getTransitionStatesAndProbs(state, action)

        sumTransitionStates = 0
        for t in transitionStates:
          #nextState, prob = t
          nextState = t[0]
          prob = t[1]
          print "\t", prob
          sumTransitionStates += prob*self.getValue(nextState)

        print 40*"*" 
        print sumTransitionStates
        #print self.mdp.getReward(state, action, nextState) + self.discount*sumTransitionStates

        return self.mdp.getReward(state, action, nextState) + self.discount*sumTransitionStates


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        actions = self.mdp.getPossibleActions(state)
        
        maxActionScore= -float('Inf')
        bestAction = None
        for a in actions:
          transitionStates = self.mdp.getTransitionStatesAndProbs(state, a)
          
          transitionStateScore = 0
          for t in transitionStates:
            
            nextState = t[0]
            prob = t[1]
            transitionStateScore += prob * self.getValue(nextState)

          if transitionStateScore > maxActionScore:
              maxActionScore = transitionStateScore
              bestAction = a


    
        if(len(actions)==0):
          return None
        else:
          return bestAction


        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
