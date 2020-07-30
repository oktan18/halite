from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

# Create a test environment for use later
board_size = 20
environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
agent_count = 4
environment.reset(agent_count)
state = environment.state[0]
board = Board(state.observation, environment.configuration)
print(board)
from preprocess import *
print(sum(all_board_params(board)[2]))
print(all_board_params(board)[2])

