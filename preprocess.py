from kaggle_environments.envs.halite.helpers import *
import json
import numpy as np


def all_board_params(board):
    board_size = int(np.sqrt(len(board._cells.items())))
    np_board_halite = np.zeros((board_size, board_size))
    np_board_ships = np.zeros((board_size, board_size))
    np_board_ships_owners = np.zeros((board_size, board_size))
    np_board_shipyard = np.zeros((board_size, board_size))
    np_board_shipyard_owners = np.zeros((board_size, board_size))
    for p, val in board._cells.items():
        np_board_halite[board_size-p[1]-1, p[0]] = val._halite
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
        np_board_ships[board_size-p[1]-1, p[0]] = ship
        np_board_ships_owners[board_size-p[1]-1, p[0]] = ship_owner
        np_board_shipyard[board_size-p[1]-1, p[0]] = shipyard
        np_board_shipyard_owners[board_size-p[1]-1, p[0]] = shipyard_owner
    return np_board_halite, np_board_ships + 1, np_board_ships_owners,\
           np_board_shipyard + 1, np_board_shipyard_owners + 1


def get_data(board, point, add_lst, part_size=5):
    (
        np_board_halite, np_board_ships, np_board_ships_owners,
        np_board_shipyard, np_board_shipyard_owners
    ) = all_board_params(board)

    a = np.concatenate([get_npboard_part_by_point(x, point, part_size) for x in [
        np_board_halite,
        # np_board_ships,
        np_board_ships_owners,
        # np_board_shipyard,
        np_board_shipyard_owners
    ]])
    return np.concatenate([a.reshape(1, part_size * part_size * 3), np.array([add_lst])], axis=1)


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

    res = big_board[y_min: y_max, x_min: x_max]
    return res