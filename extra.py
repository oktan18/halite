from kaggle_environments.envs.halite.helpers import *
import json
import numpy as np

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

X_train = []
Y_train = []

SHIP_ACTS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, ShipAction.CONVERT, None]

def all_board_params(board):
    board_size = int(np.sqrt(len(board._cells.items())))
    np_board_halite = np.zeros((board_size, board_size))
    np_board_ships = np.zeros((board_size, board_size))
    np_board_ships_owners = np.zeros((board_size, board_size))
    np_board_shipyard = np.zeros((board_size, board_size))
    np_board_shipyard_owners = np.zeros((board_size, board_size))
    for p, val in board._cells.items():
        np_board_halite[p] = val._halite
        ship_id = val._ship_id
        shipyard_id = val._shipyard_id
        if ship_id:
            ship, ship_owner = [int(x) for x in ship_id.split('-')]
        else:
            ship_owner, ship = 0, -1

        if shipyard_id:
            shipyard, shipyard_owner = [int(x) for x in shipyard_id.split('-')]
        else:
            shipyard, shipyard_owner = -1, -1
        np_board_ships[p] = ship
        np_board_ships_owners[p] = ship_owner
        np_board_shipyard[p] = shipyard
        np_board_shipyard_owners[p] = shipyard_owner
    return np_board_halite, np_board_ships + 1, np_board_ships_owners,\
           np_board_shipyard + 1, np_board_shipyard_owners + 1

def get_data(board, point, ship_halite, full_halite, part_size=5):
    (
        np_board_halite, np_board_ships, np_board_ships_owners,
        np_board_shipyard, np_board_shipyard_owners
    ) = all_board_params(board)

    a = np.concatenate([get_npboard_part_by_point(x, point, part_size) for x in [
        np_board_halite,
        np_board_ships,
        np_board_ships_owners,
        np_board_shipyard,
        np_board_shipyard_owners
    ]])
    return np.concatenate([a.reshape(1, part_size * part_size * 5), np.array([[ship_halite, full_halite]])], axis=1)


# Returns the commands we send to our ships and shipyards
def agent(obs, config):
    global X_train, Y_train
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
        X_train += [data, ]
        if ship.next_action is None:

            # Part 1: Set the ship's state
            if ship.halite < 200:  # If cargo is too low, collect halite
                ship_states[ship.id] = "COLLECT"
            if ship.halite > 500:  # If cargo gets very big, deposit halite
                ship_states[ship.id] = "DEPOSIT"

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
        Y_train += [i for i in range(len(SHIP_ACTS)) if SHIP_ACTS[i] == ship.next_action]

        with open('data/base/x.json', 'w') as f:
            json.dump([x.tolist() for x in X_train], f)

        with open('data/base/y.json', 'w') as f:
            json.dump(Y_train, f)

    return me.next_actions


def get_npboard_part_by_point(npboard, point, part_size):
    x = point[0]
    y = npboard.shape[0] - point[1] - 1
    half_part_size = part_size // 2
    board_size = npboard.shape[0]
    big_board = np.concatenate([npboard, npboard, npboard], axis=0)
    big_board = np.concatenate([big_board, big_board, big_board], axis=1)

    x_min = board_size + x - half_part_size
    x_max = board_size + x + half_part_size + 1
    y_min = board_size + y - half_part_size
    y_max = board_size + y + half_part_size + 1

    res = big_board[x_min: x_max, y_min: y_max]
    return res
