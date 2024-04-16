# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import CONTROL  
import pre_and_supporter as pas 
from pre_and_supporter import goal_model 
# import tensorflow as tf 


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    ##@tf.function  # Decorate functions for potential GPU execution  # Decorate functions for potential GPU execution
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    # #@tf.function  # Decorate functions for potential GPU execution
    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance

#@tf.function  # Decorate functions for potential GPU execution
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


#@@@-----------------------------------------------------------------------------------------------------------------------------
#!!! CODED HERE 
class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    ##@tf.function  # Decorate functions for potential GPU execution  # Decorate functions for potential GPU execution
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        CONTROL.NUM_TURN_GAME += 1 
        if(CONTROL.USE_NEW_MODE): 
            goal_model.define_goal(state=gameState) 
        #util.raiseNotDefined()
        def minimax(state):
            bestValue, bestAction = None, None
            if(CONTROL.ALLOW_LOGGING): 
                print(state.getLegalActions(0))
            else: 
                CONTROL.log_str += str(state.getLegalActions(0)) + "\n"
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            if(CONTROL.ALLOW_LOGGING): 
                print(value)
            else: 
                CONTROL.log_str += str(value) + "\n"
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth):
            #!!! when call 
            if depth > self.depth:
                return self.evaluationFunction(state)
            
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = minimax(gameState)
        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)
        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))

        # return result        

#!!! CODED HERE 
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    ##@tf.function  # Decorate functions for potential GPU execution  # Decorate functions for potential GPU execution
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        #!!! 
        "*** YOUR CODE HERE ***"
        CONTROL.NUM_TURN_GAME += 1 
        if(CONTROL.USE_NEW_MODE): 
            goal_model.define_goal(state=gameState) 
        # util.raiseNotDefined()
        def alphaBeta(state, alpha, beta):
            """
            This function implements the Alpha-Beta Pruning algorithm to determine the best action 
            for a player in a given game state.
            
            Args:
            - state: The current game state.
            - alpha: The current best value achievable by the maximizing player.
            - beta: The current best value achievable by the minimizing player.
            
            Returns:
            - bestAction: The best action to be taken by the maximizing player.
            """
            bestValue, bestAction = None, None #*** find the max value and action, # Initializing the best value and action
            if(CONTROL.ALLOW_LOGGING): 
                print(state.getLegalActions(0)) # Printing legal actions if logging is allowed
            else: 
                CONTROL.log_str += str(state.getLegalActions(0)) + "\n"
            value = [] # List to store the values of successor states
            for action in state.getLegalActions(0):
                #? if(action == 'Stop'): continue 
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1, alpha, beta) #*** next to turns of Ghosts, # Evaluating successor states
                value.append(succ) # Appending the value of successor state
                if bestValue is None: # If bestValue is not set
                    bestValue = succ # Setting bestValue to the first successor value
                    bestAction = action # Setting bestAction to the corresponding action
                else:
                    if succ > bestValue: # If the successor value is better than the current best value
                        bestValue = succ # Update bestValue
                        bestAction = action # Update bestAction
            if(CONTROL.ALLOW_LOGGING): 
                print(value) # Printing values if logging is allowed
            else: 
                CONTROL.log_str += str(value) + "\n"
            return bestAction  # Returning the best action


        def minValue(state, agentIdx, depth, alpha, beta):
            """
            This function calculates the minimum value achievable by the minimizing player (Ghost)
            considering the possible successor states.

            Args:
            - state: The current game state.
            - agentIdx: The index of the current agent.
            - depth: The depth of the search in the game tree.
            - alpha: The current best value achievable by the maximizing player (Pacman).
            - beta: The current best value achievable by the minimizing player (Ghost).

            Returns:
            - value: The minimum value achievable by the minimizing player.
            """
            if agentIdx == state.getNumAgents(): # If all agents have taken their turn
                return maxValue(state, 0, depth + 1, alpha, beta) #*** next to turn of Pacman, # Move to the next turn of Pacman
            value = None # Initializing value
            for action in state.getLegalActions(agentIdx): # Loop through legal actions for the current agent
                # Recursively calculate min value
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha=alpha, beta=beta) #*** go to turn of next Ghost, so it call Min
                if value is None: 
                    value = succ
                else:
                    value = min(value, succ) # Update value to minimum of current value and successor value
                    
                #*** prunning alpha beta           
                if value <= alpha: # If current value is less than or equal to alpha
                    return value # Prune the branch
                beta = min(beta, value) # Update beta with minimum of current beta and value
                
            #*** if it is the end state then return evalation
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth, alpha, beta):
            """
            This function calculates the maximum value achievable by the maximizing player (Pacman)
            considering the possible successor states.

            Args:
            - state: The current game state.
            - agentIdx: The index of the current agent.
            - depth: The depth of the search in the game tree.
            - alpha: The current best value achievable by the maximizing player (Pacman).
            - beta: The current best value achievable by the minimizing player (Ghost).

            Returns:
            - value: The maximum value achievable by the maximizing player.
            """
            #*** limit the depth of look search 
            #!!! when call evaluate 
            if depth > self.depth: 
                return self.evaluationFunction(state)
            
            value = None # Initializing value
            for action in state.getLegalActions(agentIdx): # Loop through legal actions for the current agent
                # Recursively calculate min value
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha=alpha, beta=beta) #*** next to turn of Ghosts 
                if value is None:
                    value = succ
                else:
                    value = max(value, succ) # Update value to maximum of current value and successor value

                #*** prunning alpha beta  
                if value >= beta: # If current value is greater than or equal to beta
                    return value # Prune the branch
                alpha = max(alpha, value) # Update alpha with maximum of current alpha and value
            
            #*** if it is the end state then return evalation
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphaBeta(gameState, -1e9, +1e9) # Calling the alpha-beta function to get the best action
        return action # Returning the best action


#!!! CODED HERE 
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    #@tf.function  # Decorate functions for potential GPU execution  # Decorate functions for potential GPU execution
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        #!!! 
        "*** YOUR CODE HERE ***"
        CONTROL.NUM_TURN_GAME += 1 
        if(CONTROL.USE_NEW_MODE): 
            goal_model.define_goal(state=gameState) 
        # util.raiseNotDefined()
        def expectedMax(state):
            """
            This function implements the Expected Maximum algorithm to determine the best action 
            for a player in a given game state.

            Args:
            - state: The current game state.

            Returns:
            - bestAction: The best action to be taken by the player.
            """
            bestValue, bestAction = None, None  # Initializing the best value and action
            if CONTROL.ALLOW_LOGGING:
                print(state.getLegalActions(0))  # Printing legal actions if logging is allowed
            else: 
                CONTROL.log_str += str(state.getLegalActions(0)) + "\n"
            value = []  # List to store the expected values of successor states
            for action in state.getLegalActions(0):
                succ = EValue(state.generateSuccessor(0, action), 1, 1)  # Calculating expected value of successor states
                value.append(succ)  # Appending the expected value
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            if CONTROL.ALLOW_LOGGING:
                print(value)  # Printing expected values if logging is allowed
            else: 
                CONTROL.log_str += str(value) + "\n"
            return bestAction


        def EValue(state, agentIdx, depth):
            """
            This function calculates the expected value of the successor states.

            Args:
            - state: The current game state.
            - agentIdx: The index of the current agent.
            - depth: The depth of the search in the game tree.

            Returns:
            - The expected value of the successor states.
            """
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)  # Move to the next turn of Pacman
            value = []  # List to store the values of successor states
            for action in state.getLegalActions(agentIdx):
                succ = EValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)  # Recursively calculate expected value
                value.append(succ)

            if len(value) != 0:
                return 1.0 * sum(value) / len(value)  # Calculate the average expected value
            else:
                return self.evaluationFunction(state)  # Return evaluation if no successor state


        def maxValue(state, agentIdx, depth):
            """
            This function calculates the maximum value achievable by the maximizing player.

            Args:
            - state: The current game state.
            - agentIdx: The index of the current agent.
            - depth: The depth of the search in the game tree.

            Returns:
            - The maximum value achievable by the maximizing player.
            """
            #!!! when call evaluation 
            if depth > self.depth:
                return self.evaluationFunction(state)  # Return evaluation if depth limit reached
            
            value = None  # Initializing value
            for action in state.getLegalActions(agentIdx):
                succ = EValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)  # Calculate expected value
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)  # Update value to maximum of current value and successor value
                
            if value is not None:
                return value  # Return the maximum value
            else:
                return self.evaluationFunction(state)  # Return evaluation if no successor state

        action = expectedMax(gameState)  # Calling the Expected Maximum algorithm to get the best action
        return action  # Returning the best action
        
        
#!!! DOING HERE 
#@tf.function  # Decorate functions for potential GPU execution
def betterEvaluationFunction(currentGameState):
    """
    This evaluation function aims to improve the performance of the Pacman agent by considering various factors 
    such as the position of Pacman, the layout of the food pellets, the positions of the ghosts, the availability 
    of capsules, and the scared timer of ghosts.

    DESCRIPTION:
    - Calculate the position of Pacman, food pellets, ghosts, capsules, and scared timer.
    - Determine the closest distance to ghosts and capsules, and calculate their respective values.
    - Evaluate the distance to the closest food pellet and calculate its value.
    - Combine these factors to generate an evaluation score.

    Returns:
    - The evaluation score for the given game state.
    """
    "*** YOUR CODE HERE ***"
    if(CONTROL.SCORE_MODE): 
        return scoreEvaluationFunction(currentGameState)
    
    if(CONTROL.USE_NEW_MODE): 
        return goal_model_betterEvaluationFunction(currentGameState=currentGameState) 
        
    #** atribute used to compuet and control 
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #*** freeze Ghosts 
    '''
    example for params: 
    --- Game is start, ---
    (6, 1)
    FFFFFFFFFFFFFFFFFFFF ---> asList(): list of positions are true 
    FFTTTFTTTTTTTTFTTTTF
    FTFFTFTFFFFFFTFTFFTF
    FTFTTTTTTTTTTTTTTFTF
    FTFTFFTFFFFFFTFFTFTF
    FTTTTTTFFFFFFTTTTTTF
    FTFTFFTFFFFFFTFFTFTF
    FTFTTTTTTTTTTTTTTFTF
    FTFFTFTFFFFFFTFTFFTF
    FTTTTFFFFFTTTTFTTTFF
    FFFFFFFFFFFFFFFFFFFF
    [<game.AgentState object at 0x0000017DD5F35B10>, <game.AgentState object at 0x0000017DD5F35B50>] ---> getPosition() 
    [(18, 1), (1, 9)]
    [0, 0]
    '''
    """
    if(CONTROL._PRINT_EXAMPLE > 0): 
        print("\n---Example of params in function, multiAgents/betterEvalFn()---")
        print(newPos)
        print(newFood)
        print(newGhostStates)
        print(newCapsules)
        print(newScaredTimes)
        CONTROL._PRINT_EXAMPLE -= 1 
        print("CONTROL._PRINT_EXP change =>", CONTROL._PRINT_EXAMPLE, "\n") 
`   """
    # Calculate the closest distance to a ghost
    closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    # Calculate the closest distance to a capsule
    if newCapsules:
        closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
    else:
        closestCapsule = 0
    """evaluate"""
    # Evaluate the value of the closest capsule
    if closestCapsule:
        closest_capsule = -3 / closestCapsule
    else:
        closest_capsule = 100
    # Evaluate the value of the distance to the closest ghost
    if closestGhost:
        ghost_distance = -2 / closestGhost
    else:
        ghost_distance = -500
    """Food as List"""
    # Calculate the distance to the closest food pellet
    foodList = newFood.asList()
    if foodList:
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    else:
        closestFood = 0
    # Combine factors to generate evaluation score
    return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule


import math 
# This function aims to improve the performance of the Pacman agent by considering various factors 
# such as the position of Pacman, the layout of the food pellets, the positions of the ghosts, the availability 
# of capsules, and the scared timer of ghosts.
'''
Docstring:

    Provides a good explanation of the function's purpose and the factors considered for evaluation.
    Function Arguments:

    currentGameState: An object representing the current state of the Pacman game.
    Retrieving Game State Information:

    pos: Pacman's current position.
    foods: List of positions for all remaining food pellets.
    ghosts: List of positions for all ghosts.
    capsus: List of positions for all capsules (power pellets).
    newScaredTimes: List of scared timer values for each ghost (higher value means ghost is scared for longer).
    Calling External Functions (Likely from goal_model):

    These functions seem to be responsible for calculating penalty and bonus scores based on various factors.
    pen_goal: Penalty score related to the overall goal (unclear from the provided code).
    pen_total_with_capsus: Penalty score considering capsules.
    pen_total_with_ghosts: Penalty score considering ghosts and their scared timers.
    bonus_foods: Bonus score based on the number of food pellets.
    bonus_capsu: Bonus score based on the number of capsules.
    bonus_freeze: Bonus score related to scared ghosts (possibly based on proximity).
    bonus_score: Bonus score based on a potentially custom calculation.
    get_scale_param: Function to adjust the scale of specific scores (if training is not enabled).
    Scaling Scores (if Training is not Active):

    Scales the penalty and bonus scores retrieved from external functions, possibly based on learned parameters (unclear without the goal_model code).
    Endgame Bonus:

    Increases the gscore (overall score) based on the number of remaining food pellets, incentivizing collecting all food before ending the game.
    Scared Ghost Bonus:

    Increases the g3 bonus score (possibly related to eating ghosts) if Pacman is close enough to a scared ghost.
    Applying Weights and Normalization:

    Applies weights from CONTROL class (likely containing constants) to various scores (e.g., rate_g1 for food weight, rate_f3 for ghost distance weight).
    Normalizes the penalty and bonus scores by dividing each by a constant term (potentially for better balancing).
    Calculating Additional Scores:

    gwin: Set to CONTROL.rate_gwin if all food is eaten (win bonus).
    fover: Set to CONTROL.rate_fover if Pacman overlaps with a ghost (penalty for being close).
    Final Score Calculation:

    Returns a weighted sum of normalized bonus scores, negative normalized penalty scores, win/overlap penalties/bonuses, and overall game score.
    Overall Comments:

    The function uses a combination of penalties, bonuses, and weights to calculate a comprehensive evaluation score.
    The code relies on external functions from goal_model for specific calculations.
    Scaling scores based on training might indicate the use of a reinforcement learning approach (unclear without seeing goal_model).
    Consider adding comments to the external functions for better understanding.
    Additional Recommendations:

    Explore using more descriptive variable names (e.g., scared_times instead of newScaredTimes).
    If CONTROL is a class, ensure proper encapsulation and documentation of its constants.
    Consider using a logging library for detailed information about the evaluation process (optional).
'''
#@tf.function  # Decorate functions for potential GPU execution
def goal_model_betterEvaluationFunction(currentGameState):
    """
        This evaluation function aims to improve the performance of the Pacman agent by considering various factors:

        * Pacman's position
        * Food pellet layout
        * Ghost positions
        * Capsule availability
        * Ghost scared timers

        Calculates:
            - Closest distances to ghosts and capsules, with their respective values.
            - Distance to the closest food pellet and its value.
            - Combines these factors to generate an evaluation score.

        Args:
            currentGameState: Object representing the current state of the Pacman game.

        Returns:
            The evaluation score for the given game state.
    """
    "*** YOUR CODE HERE ***"
    # Get game state information
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()  # List of food positions
    ghosts = [x.getPosition() for x in currentGameState.getGhostStates()]  # List of ghost positions
    capsules = currentGameState.getCapsules()  # List of capsule positions
    new_scared_times = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]  # List of scared timer values

    # Call external functions (likely from goal_model)
    f1 = goal_model.pen_goal(currentGameState=currentGameState)  # Penalty score related to overall goal
    f2 = goal_model.pen_total_with_capsules(pos, capsules=capsules)  # Penalty score considering capsules
    f3 = goal_model.pen_total_with_ghosts(pos, ghosts=ghosts, new_scared_times=new_scared_times)  # Penalty score considering ghosts and scared timers
    g1 = goal_model.bonus_foods(foods=foods)  # Bonus score based on number of food pellets
    g2 = goal_model.bonus_capsu(capsules=capsules)  # Bonus score based on number of capsules
    g3 = goal_model.bonus_freeze(pos, ghosts=ghosts, new_scared_times=new_scared_times)  # Bonus score related to scared ghosts
    gscore = goal_model.bonus_score(currentGameState=currentGameState)  # Overall game score (custom calculation?)

    # Scale scores if training is not active (using parameters from CONTROL class)
    if not CONTROL.TRAINING:
        f1 = goal_model.get_scale_param('goal', f1)
        f2 = goal_model.get_scale_param('capsules', f2)
        f3 = goal_model.get_scale_param('ghosts', f3)
        g1 = goal_model.get_scale_param('bonus_foods', g1)
        g2 = goal_model.get_scale_param('bonus_capsu', g2)
        g3 = goal_model.get_scale_param('freeze', g3)
        gscore = goal_model.get_scale_param('score', gscore)

    # Endgame bonus based on remaining food pellets
    if len(foods) < CONTROL.ENDGAME_FOOD_COUNT:
        gscore *= (CONTROL.ENDGAME_FOOD_COUNT - len(foods))
    else:
        gscore *= CONTROL.RATE_SCORE  # Weight for overall game score

    # Scared ghost bonus if Pacman is close enough
    d_ghost = goal_model.real_distance_freeze_ghost(pos, ghosts, new_scared_times)  # Distance to closest scared ghost
    if d_ghost < CONTROL.EAT_GHOST_DISTANCE:
        g3 *= CONTROL.RATE_EAT_3_GHOSTS  # Weight for eating ghosts

    # Apply weights and normalization
    g1 *= CONTROL.RATE_EAT_FOOD  # Weight for food score
    f3 *= CONTROL.RATE_GHOST  # Weight for ghost distance penalty
    norm_f = 1.0  # Normalization factor (penalty scores)
    norm_g = 1.0  # Normalization factor (bonus scores)
    
    gwin, fover = 0, 0 
    if len(foods) == 0: 
        gwin = CONTROL.RATE_WIN
    for ghost in ghosts: 
        if ghost == pos: 
            fover = CONTROL.RATE_OVER_GAME 
        
    # Combine factors to generate the evaluation score
    return g1 + g2 + g3 - f1 - f2 - f3 - fover + gwin + gscore 

# Abbreviation
better = betterEvaluationFunction # not effect 

