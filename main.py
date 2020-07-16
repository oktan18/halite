from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

import numpy as np
import keras

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


def get_move(board, point, model, part_size=5):
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
    step = SHIP_ACTS[np.argmax(model.predict(a.reshape(1, part_size*part_size*5)))]
    return step


def eval_model(model, board_size=20):
    environment = make("halite", configuration={"size": board_size, "startingHalite": 1000}, debug=True)
    environment.run([agent(model), "random", "random", "random"])
    return eval_env(environment)


def fit(nets):
    c = 0
    for net in nets:
        c += 1
        if c % 5==0:
            print('    ', c)
        full_res = []
        for _ in range(3):
            res = eval_model(net.model())
            full_res += [res, ]
        score = np.array(full_res).mean(axis=0)[0]
        wins = np.mean(np.array(full_res)[:, 0] > 0)
        net.score = score
        net.wins = wins
    return nets


class Net:
    def __init__(self, part_size):


        self.part_size = part_size

        self._acts = np.random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'], 3)
        self._w = None

    def init_hyperparams(self):
        hyperparams = {
            'acts': self._acts,
            'w': self._w,
        }
        return hyperparams

    def model(self):
        inputs = keras.Input(shape=(self.part_size * self.part_size * 5,), name="digits")
        x = keras.layers.Dense(128, activation=self._acts[0], name="dense_1", kernel_initializer='random_normal',
                               bias_initializer='zeros')(inputs)
        x = keras.layers.Dense(32, activation=self._acts[1], name="dense_2", kernel_initializer='random_normal',
                               bias_initializer='zeros')(x)
        x = keras.layers.Dense(32, activation=self._acts[2], name="dense_3", kernel_initializer='random_normal',
                               bias_initializer='zeros')(x)
        outputs = keras.layers.Dense(6, activation="softmax", name="predictions", kernel_initializer='random_normal',
                                     bias_initializer='zeros')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        if self._w is None:
            self._w = [x.numpy() for x in model.weights]
        model.set_weights(self._w)
        return model

    def play(self, board_size):
        environment = make("halite", configuration={"size": board_size, "startingHalite": 1000}, debug=True)
        environment.run([agent(self.model()), "random", "random", "random"])
        environment.render(mode="ipython", width=800, height=600)


def agent(model):
    def agent_m(model, observation, configuration):
        board = Board(observation, configuration)
        me = board.current_player
        if len(me.ships) == 0 and len(me.shipyards) > 0:
            me.shipyards[0].next_action = ShipyardAction.SPAWN

        # If there are no shipyards, convert first ship into shipyard.
        if len(me.shipyards) == 0 and len(me.ships) > 0:
            me.ships[0].next_action = ShipAction.CONVERT

        for ship in me.ships:
            if ship.next_action == None:
                step = get_move(board, ship.position, model)
                ship.next_action = step
        return me.next_actions
    return lambda observation, configuration: agent_m(model, observation, configuration)


def eval_env(env):
    res = [x[0] for x in env.steps[-1][0]['observation']['players']]
    r = []
    for i in range(len(res)):
        score = res[i]
        max_other = max([res[j] for j in range(len(res)) if i != j])
        r += [score-max_other]
    return r
