from kaggle_environments.envs.halite.helpers import *
import json
import numpy as np

from preprocess import get_npboard_part_by_point, get_data


def getDirTo(fromPos, toPos, size):
    fromX, fromY = divmod(fromPos[0], size), divmod(fromPos[1], size)
    toX, toY = divmod(toPos[0], size), divmod(toPos[1], size)
    if fromY < toY: return ShipAction.NORTH
    if fromY > toY: return ShipAction.SOUTH
    if fromX < toX: return ShipAction.EAST
    if fromX > toX: return ShipAction.WEST


# Directions a ship can move
directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]

# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard
ship_states = {}


# Returns the commands we send to our ships and shipyards
def agent(obs, config):
    size = config.size
    board = Board(obs, config)
    me = board.current_player

    # If there are no ships, use first shipyard to spawn a ship.
    if len(me.ships) == 0 and len(me.shipyards) > 0:
        me.shipyards[0].next_action = ShipyardAction.SPAWN

    # If there are no shipyards, convert first ship into shipyard.
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT

    for ship in me.ships:
        data = get_data(board, ship.position, ship.halite, me.halite)
        if ship.next_action is None:

            # Part 1: Set the ship's state
            if ship.halite < 200:  # If cargo is too low, collect halite
                ship_states[ship.id] = "COLLECT"
            if ship.halite > 500:  # If cargo gets very big, deposit halite
                ship_states[ship.id] = "DEPOSIT"
                y = [6, ]

            # Part 2: Use the ship's state to select an action
            if ship_states[ship.id] == "COLLECT":
                # If halite at current location running low,
                # move to the adjacent square containing the most halite
                if ship.cell.halite < 100:
                    neighbors = [ship.cell.north.halite, ship.cell.east.halite,
                                 ship.cell.south.halite, ship.cell.west.halite]
                    best = max(range(len(neighbors)), key=neighbors.__getitem__)
                    ship.next_action = directions[best]
            if ship_states[ship.id] == "DEPOSIT":
                # Move towards shipyard to deposit cargo
                direction = getDirTo(ship.position, me.shipyards[0].position, size)
                if direction:
                    ship.next_action = direction

        # with open('data/base/x.json', 'w') as f:
        #     json.dump([x.tolist() for x in X_train], f)
        #
        # with open('data/base/y.json', 'w') as f:
        #     json.dump(Y_train, f)

    return me.next_actions


