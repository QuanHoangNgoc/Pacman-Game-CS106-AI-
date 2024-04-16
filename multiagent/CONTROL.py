"""
Configuration Variables:

ALLOW_LOG (bool): Controls whether logging is enabled.
GRAPHIC (bool): Controls whether graphics are displayed (likely during gameplay).
SCORE_MODE (bool): Purpose unclear from the comment.
USE_NEW_MODE (bool): Enables a new mode (details unknown).
TOP_NN (int): Number of nearest neighbors to consider (possibly for evaluation).
PATIENT (int): Patience parameter (usage unclear without context).
Ghost Penalty Mode:

FREEZE_GHOST_MODE_PEN (bool): If True, ignores penalized ghosts in calculations (comment unclear on "con nao bi dong bang").
DYNAMIC_GHOST_PEN (bool): Controls static or dynamic penalty for ghosts (comment needs clarification).
DYNAMIC_GHOST_BONUS (bool): Enables dynamic ghost bonus (details unknown).
Game Parameters:

EAT_GHOST_DIS (int): Minimum distance required to eat a ghost.
rate_g3 (float): Reward for eating 3 ghosts (unclear unit).
ENDGAME_FOOD (int): Number of food items remaining at game end.
rate_gscore (float): Weight for ghost score in evaluation (unclear range).
rate_gwin (float): Reward for winning the game (unclear unit).
rate_fover (float): Reward for having more food than ghosts (unclear unit).
rate_g1 (float): Weight for single ghost in evaluation (unclear range).
rate_f3 (float): Reward for eating 3 food items (unclear unit).
Logging and File Handling:

log_str (str): Empty string, likely intended for log messages (unused).
Imports: Standard library modules for randomness, date/time, and file operations.
current_time (datetime.time): Gets the current time (only the hour and minute).
_strrd (str): Creates a timestamp string with hour, minute, and random number (unclear purpose).
os.makedirs('LOG', exist_ok=True): Creates the LOG directory if it doesn't exist.
LOG_FILE: Opens a new log file with the timestamp in the LOG directory.
OPEN_FILE: Opens a file named 1010_open_file.txt in append mode (purpose unclear).
Training and Game Parameters:

TRAINING (bool): Controls whether training mode is active.
NUM_TURN_GAME (int): Number of turns in a game (possibly uninitialized).
NUM_GAME (int): Total number of games to play (set to 7).
NUM_TRAIN (int): Number of training games (set to 2).
Git Commands (Commented Out):

These commands demonstrate how to run the program with different agents and parameters, as well as basic Git operations. However, using git push --force is generally discouraged as it rewrites history on the remote repository.
Overall Comments:

Variable names could be more descriptive (e.g., allow_logging instead of ALLOW_LOG).
Comments could be improved to provide clearer explanations.
Some constants lack context or units for proper understanding.
Consider using a separate configuration file or module for better organization.
"""

##############################################################################################
# Configuration for running the Pacman game.
##############################################################################################
# Enable or disable logging (currently disabled).
ALLOW_LOGGING = False
# Enable or disable graphics for the game (currently disabled).
GRAPHICS_ENABLED = False

# Flag for a specific scoring method (purpose unclear).
SCORE_MODE = False
# Enable a new mode (details unknown from the code).
USE_NEW_MODE = True
# Number of training games.
NUM_TRAIN = 2
# Total number of games to play.
NUM_GAMES = NUM_TRAIN + 1  # Consider using a clearer calculation


##############################################################################################
# Mode for calculating ghost penalty:
##############################################################################################
# If True, ignores penalized ghosts in calculations.
# If False, considers both penalized and non-penalized ghosts.
FREEZE_GHOST_MODE_PENALTY = True  # More descriptive name
# Controls static or dynamic penalty for ghosts:
# False: Static penalty (details unknown).
DYNAMIC_GHOST_PENALTY = False
# Enables dynamic ghost bonus (details unknown).
DYNAMIC_GHOST_BONUS = False


##############################################################################################
# Game Parameters:
##############################################################################################
# Number of nearest neighbors to consider for evaluation (e.g., nearest food or ghosts).
TOP_NEIGHBORS = 3
# Patience parameter (usage unclear without context).
PATIENCE = 2
# Minimum distance required to eat a ghost (presumably in grid units).
EAT_GHOST_DISTANCE = 11
# Reward for eating 3 ghosts (unit could be points or score).
RATE_EAT_3_GHOSTS = 3.0
# Number of food items remaining at game end (indicates a win?).
ENDGAME_FOOD_COUNT = 5
# Weight for ghost score in evaluation (likely between 0 and 1).
RATE_SCORE = 0.25

# Reward for winning the game (unit could be points or score).
RATE_WIN = 10
# Reward for having more food than ghosts (unit could be points or score).
RATE_OVER_GAME = 20
# Weight for a single ghost in evaluation (likely between 0 and 1).
RATE_GHOST = 1.5 
# Reward for eating food items 
RATE_EAT_FOOD = 0.75 


##############################################################################################
# Logging and File Handling (Improved with Error Handling):
##############################################################################################
# Empty string for potential log messages.
log_str = ""
import random
import datetime
import os
ID_LOG_FILE = str(random.randint(1, 100))
def create_log_file():
    """Creates a new log file with a timestamp."""
    current_time = datetime.datetime.now().time()
    timestamp = current_time.strftime('%H-%M')  # No need for random number
    global ID_LOG_FILE
    timestamp = timestamp + "=" + ID_LOG_FILE 
    ID_LOG_FILE = timestamp 
    
    try:
        # Create the LOG directory if it doesn't exist.
        os.makedirs('LOG', exist_ok=True)

        # Open a new log file with the timestamp in the LOG directory.
        filename = f'log_file_{timestamp}.txt'
        global LOG_FILE
        LOG_FILE = open(os.path.join('LOG', filename), 'w')
    except OSError as e:
        print(f"Error creating log file: {e}")

# Open the log file (call the function to ensure it's created)
create_log_file()
# File for unknown purpose (consider removing or investigating).
OPEN_FILE = open('1010_open_file.txt', 'a')
TRAINING = False
# Number of turns in a game (possibly uninitialized).
NUM_TURNS_GAME = 0  # Corrected variable name



'''
python pacman.py -l mediumClassic -p MinimaxAgent -a depth=3,evalFn=betterEvaluationFunction -s 22521178 
python pacman.py -l mediumClassic -p ExpectimaxAgent -a depth=2,evalFn=betterEvaluationFunction -s 22521178 
python pacman.py -l mediumClassic -p ExpectimaxAgent -a depth=3,evalFn=betterEvaluationFunction -s 22521178 
git init
git status
git add . 
git commit -m "[folder upload -force]" 
git branch -M main 
git remote add origin https://github.com/QuanHoangNgoc/Container.git
git push --force origin main
'''
