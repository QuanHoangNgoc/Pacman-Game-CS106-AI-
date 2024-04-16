# Import necessary libraries
from collections import defaultdict
import util

# Define constant for wall
WALL = '%'

##############################################################################################
# Pre Building Dijkstra for compute distance 
##############################################################################################
# Function to convert layout into map array
def _get_arr_map(layout): 
  map_str = str(layout)
  words = map_str.split('\n')
  lx, ly = len(words[0]), len(words) 

  # Initialize defaultdict with WALL as default value
  a = defaultdict(lambda x: WALL)
  rid = 0  
  for y in range(ly-1, -1, -1): 
    for x in range(0, lx): 
      while(map_str[rid]=='\n'): rid += 1
      a[(x, y)] = map_str[rid] 
      rid += 1 
  return a, lx, ly 

# Function to perform Dijkstra's algorithm from one point to all other points
def _dijkstra_one_to_all(sx, sy, arr, full_distance): 
  q = util.Queue()
  
  # Set distance from starting point to itself as 0
  full_distance[sx, sy, sx, sy] = 0  
  q.push((sx, sy))
  
  # Define directional increments
  dx = [-1, +1, 0, 0]
  dy = [0, 0, -1, +1]
  
  # Perform Dijkstra's algorithm
  while(q.isEmpty() == False):
    u, v = q.pop()
    duv = full_distance[(sx, sy, u, v)]
    
    for t in range(4): 
      x, y = u + dx[t], v + dy[t] 
      dxy = full_distance[(sx, sy, x, y)]
      
      # Update distance if it's shorter than the current one
      if(arr[x, y] != WALL and dxy > duv + 1): 
        full_distance[sx, sy, x, y] = duv + 1
        q.push((x, y))
                
# Define infinity constant
INF = 1e5 + 7 

# Function to return infinity
def return_inf(): 
  return INF

# Global variable for distance
_DISTANCE = None 

# Function to build full distance map
def build_full_distance(layout): 
  # Get map to array 
  arr, lx, ly = _get_arr_map(layout=layout) 
  full_distance = defaultdict(return_inf) 
  
  # Build full distance map
  for x in range(lx): 
    for y in range(ly): 
      if(arr[x, y] != WALL): 
        _dijkstra_one_to_all(x, y, arr, full_distance)
      
  global _DISTANCE 
  _DISTANCE = full_distance 

# Function to get distance between two points
def get_distance(sx, sy, tx, ty): 
  return _DISTANCE[sx, sy, tx, ty]

# Define function to return negative infinity
def return_inv_inf(): 
  return -return_inf() 

# Define function to return None
def return_none():
  return None 


import CONTROL 
from itertools import permutations
'''
  Penalty and Bonus Functions:

  pen_total_with_capsules: This function calculates the penalty associated with the distance to all capsules. It could be improved by considering the importance of specific capsules (e.g., closer capsules might be more valuable).
  pen_total_with_ghosts: This function calculates the penalty based on the distance to ghosts. It considers scared timers (potentially rewarding approaching scared ghosts). We could analyze how the CONTROL.DYNAMIC_GHOST_PENALTY flag affects this behavior.
  bonus_foods: This function provides a simple bonus based on the number of remaining food pellets. We could explore if a more nuanced approach (e.g., rewarding collecting high-value food first) might be beneficial.
  bonus_freeze: This function calculates a bonus based on the closest scared ghost. It considers permutations of ghost locations and scared timers. We could analyze the efficiency of using permutations for this calculation.
  bonus_score: This function simply tracks the change in the game score. We could explore if incorporating the rate of score increase would be useful.
  Overall Goal Model Logic:

  define_goal: This function defines the subgoals (food and capsules to collect) for the current state. It considers the number of remaining food items (CONTROL.TOP_NEIGHBORS) and pre-calculates distances using get_distance. We could analyze how CONTROL.TOP_NEIGHBORS affects the agent's strategy.
  _define_sub_goal: This function retrieves information about ghost positions and scared timers for later calculations.
  _init_lock: This function initializes internal variables for tracking state changes and potentially limiting goal redefinition (unlock).
  General Improvements:

  Readability: Adding more comments within the code would improve understanding, especially for complex sections like bonus_freeze.
  Efficiency: The use of permutations in bonus_freeze might be computationally expensive for a large number of ghosts. Exploring alternative approaches could be beneficial.
  Adaptability: The model relies on control flags (CONTROL.DYNAMIC_GHOST_PENALTY, CONTROL.TOP_NEIGHBORS) to influence behavior. We could explore if these flags could be learned or adapted dynamically.
  By analyzing these aspects, you can gain a deeper understanding of the Pacman AI's decision-making process and identify potential areas for improvement.
'''
class GoalModel:
    def __init__(self) -> None:
        # Initialize instance variables
        self.unlock = 1
        self.min_param = defaultdict(return_inf)  # Initialize defaultdict with return_inf as default value
        self.max_param = defaultdict(return_inv_inf)  # Initialize defaultdict with return_inv_inf as default value
        self.name_param = set()  # Initialize set to store parameter names
        # self.goal_positions = None  # Commented out; not used in current implementation
        # self.per_pos_list = None  # Commented out; not used in current implementation

    def init_catching(self):
        # Initialize catching as a defaultdict with return_none as default value
        self.catching = defaultdict(return_none)

    #@tf.function  # Decorate functions for potential GPU execution
    def _upd_param(self, name: str, value):
        # Update min and max parameters if in training mode
        if CONTROL.TRAINING:
            self.min_param[name.lower()] = min(self.min_param[name.lower()], value)
            self.max_param[name.lower()] = max(self.max_param[name.lower()], value)
            self.name_param.add(name.lower())  # Add parameter name to the set

    #@tf.function  # Decorate functions for potential GPU execution
    def get_scale_param(self, name: str, value):
        # Calculate scaled parameter value
        EPS = 1 / INF
        if name.lower() not in self.name_param or CONTROL.TRAINING:
            return RuntimeWarning()  # Return warning if parameter not found or in training mode
        return (value - self.min_param[name.lower()]) / (self.max_param[name.lower()] - self.min_param[name.lower()] + EPS)

    #@tf.function  # Decorate functions for potential GPU execution
    def define_goal(self, state):
        # Define goals based on current game state
        self._define_sub_goal(state)
        if not self._unlock_define_goal():  # If goal is not unlocked, exit
            return

        currentGameState = state
        pos_pacman = currentGameState.getPacmanPosition()
        foods = currentGameState.getFood().asList()
        foods = sorted(foods, key=lambda x: get_distance(pos_pacman[0], pos_pacman[1], x[0], x[1]))
        capsules = currentGameState.getCapsules()

        # Initialize and define goals
        self._init_lock(state, pos_pacman, foods, capsules)
        if len(foods) < CONTROL.TOP_NEIGHBORS:
            self.goal_positions = capsules + foods
        else:
            self.goal_positions = capsules + foods[:CONTROL.TOP_NEIGHBORS]
        self.per_pos_list = list(permutations(self.goal_positions))
        if CONTROL.ALLOW_LOGGING:
            print("\n---define_goal()---")
        else:
            CONTROL.log_str += "\n---define_goal()---" + "\n"

    #@tf.function  # Decorate functions for potential GPU execution
    def _define_sub_goal(self, state):
        # Define sub-goals based on current game state
        currentGameState = state
        newGhostStates = currentGameState.getGhostStates()
        ghosts = [x.getPosition() for x in currentGameState.getGhostStates()]
        new_scared_times = [ghostState.scaredTimer for ghostState in newGhostStates]
        self.ghosts = ghosts
        self.scared = new_scared_times

        if CONTROL.ALLOW_LOGGING:
            print("\n---define_sub_goal()---")
        else:
            CONTROL.log_str += "\n---define_sub_goal()---" + "\n"

    #@tf.function  # Decorate functions for potential GPU execution
    def _init_lock(self, state, pos_pacman, foods, capsules):
        # Initialize lock based on current game state
        self.unlock = CONTROL.PATIENCE
        self.num_capsus = len(capsules)
        self.num_foods = len(foods)
        self.score = state.getScore()

    #@tf.function  # Decorate functions for potential GPU execution
    def _unlock_define_goal(self):
        # Unlock goal if conditions are met
        self.unlock -= 1
        if self.unlock <= 0:
            return True
        return False

    #@tf.function  # Decorate functions for potential GPU execution
    def pen_total_with_capsules(self, pos_pacman, capsules):
        # Calculate penalty based on capsules
        if len(capsules) == 0:
            self._upd_param('capsules', 0)
            return 0

        value = 0
        for capsu in capsules:
            value = (value + get_distance(pos_pacman[0], pos_pacman[1], capsu[0], capsu[1]))
        self._upd_param('capsules', value)
        return value

    #@tf.function  # Decorate functions for potential GPU execution
    def pen_total_with_ghosts(self, pos_pacman, ghosts, new_scared_times):
        # Calculate penalty based on ghosts
        _ghosts, _scared = self.ghosts, self.scared
        if CONTROL.DYNAMIC_GHOST_PENALTY:
            _ghosts, _scared = ghosts, new_scared_times

        value, id = 0, -1
        for ghost in _ghosts:
            id += 1
            if CONTROL.FREEZE_GHOST_MODE_PENALTY and _scared[id] == 1:
                continue  # Skip frozen ghosts
            value += get_distance(pos_pacman[0], pos_pacman[1], ghost[0], ghost[1])

        self._upd_param('ghosts', 1.0 / (value + 1))
        return 1.0 / (value + 1)

    #@tf.function  # Decorate functions for potential GPU execution
    def pen_goal(self, currentGameState):
        # Calculate penalty based on goals
        pos_pacman = currentGameState.getPacmanPosition()
        foods = currentGameState.getFood().asList()
        capsules = currentGameState.getCapsules()

        exist = set(foods) | set(capsules)
        subset = exist & set(self.goal_positions)
        subset = tuple(subset)
        if self.catching[(pos_pacman, subset)] != None:
            return self.catching[(pos_pacman, subset)]

        min_value = return_inf()
        for per_pos in self.per_pos_list:
            pos_a = pos_pacman
            total = 0
            for pos_b in per_pos:
                if pos_b in exist:
                    total += get_distance(pos_a[0], pos_a[1], pos_b[0], pos_b[1])
                    pos_a = pos_b
            min_value = min(min_value, total)

        scale_min_value = min_value
        self.catching[(pos_pacman, subset)] = scale_min_value
        self._upd_param('goal', scale_min_value)
        return scale_min_value

    #@tf.function  # Decorate functions for potential GPU execution
    def bonus_capsu(self, capsules):
        # Calculate bonus based on capsules
        self._upd_param('bonus_capsu', self.num_capsus - len(capsules))
        return self.num_capsus - len(capsules)

    #@tf.function  # Decorate functions for potential GPU execution
    def bonus_foods(self, foods):
        # Calculate bonus based on foods
        self._upd_param('bonus_foods', self.num_foods - len(foods))
        return self.num_foods - len(foods)

    #@tf.function  # Decorate functions for potential GPU execution
    def bonus_freeze(self, pos_pacman, ghosts, new_scared_times):
        # Calculate bonus based on frozen ghosts
        _ghosts, _scared = self.ghosts, self.scared
        if CONTROL.DYNAMIC_GHOST_BONUS:
            _ghosts, _scared = ghosts, new_scared_times

        if sum(_scared) == 0:
            self._upd_param('freeze', 0)
            return 0

        value = self.real_distance_freeze_ghost(pos_pacman, ghosts, new_scared_times)
        self._upd_param('freeze', 1.0 / (value + 1))
        return 1.0 / (value + 1)

    #@tf.function  # Decorate functions for potential GPU execution
    def real_distance_freeze_ghost(self, pos_pacman, ghosts, new_scared_times):
        # Calculate real distance to frozen ghosts
        _ghosts, _scared = self.ghosts, self.scared
        if CONTROL.DYNAMIC_GHOST_BONUS:
            _ghosts, _scared = ghosts, new_scared_times

        value = return_inf()
        for ghost_list, scared_list in zip(permutations(_ghosts), permutations(_scared)):
            total, cnt = 0, 0
            for ghost, scared in zip(ghost_list, scared_list):
                if scared == 0:
                    continue  # Skip non-frozen ghosts
                total += get_distance(pos_pacman[0], pos_pacman[1], ghost[0], ghost[1])
                cnt += 1
                if cnt >= 3:
                    break
            value = min(value, total)
        return value

    #@tf.function  # Decorate functions for potential GPU execution
    def bonus_score(self, currentGameState):
        # Calculate bonus based on score
        self._upd_param('score', currentGameState.getScore() - self.score)
        return currentGameState.getScore() - self.score


# Create an instance of GoalModel
goal_model = GoalModel()








  
  