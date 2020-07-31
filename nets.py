import keras
from kaggle_environments.envs.halite.helpers import *

from base import BaseShipyardManager
import numpy as np
from preprocess import *


def get_nearest_shipyard(ship, board):
    def distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_shipyards():
        me = board.current_player
        return [shipyard.position for shipyard in me.shipyards]

    ship_point = ship.position
    shipyards = get_shipyards()
    nearest_shipyard = shipyards[np.argmin([distance(ship_point, s) for s in shipyards])]
    return nearest_shipyard


class BaseNet:
    def __init__(self, part_size, mutable=True):
        self.model = None
        self.part_size = part_size
        self.init_model()
        self._weights = [x.numpy() for x in self.model.weights]
        self._scores = []
        self.mutable = mutable

    def init_model(self):
        raise NotImplementedError

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, w):
        self._weights = w
        self.model.set_weights(w)

    def mutate(self):
        if self.mutable:
            child = self.__class__(self.part_size)
            w = [wi + np.random.normal(size=wi.shape)*(np.random.random(wi.shape) < 0.05) for wi in self.weights]
            child.weights = w
            return child
        else:
            return deepcopy(self)


class ShipStateNet(BaseNet):
    def __init__(self, part_size):

        self.states = ["CONVERT", "COLLECT", "DEPOSIT"]
        super().__init__(part_size)

    def init_model(self):
        inputs = keras.Input(shape=(self.part_size * self.part_size * 3 + 4,), name="digits")
        x = keras.layers.Dense(16, activation="tanh", name="dense_1", kernel_initializer='random_normal',
                               bias_initializer='random_normal')(inputs)
        x = keras.layers.Dense(16, activation="relu", name="dense_4", kernel_initializer='random_normal',
                               bias_initializer='random_normal')(x)
        outputs = keras.layers.Dense(3, activation="softmax", name="predictions",
                                     kernel_initializer='random_normal',
                                     bias_initializer='random_normal')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def state(self, ship, board):

        me = board.current_player
        if len(me.shipyards) == 0:
            return "CONVERT"

        me = board.current_player
        nearest_shipyard = get_nearest_shipyard(ship, board)

        inpt = get_data(board, ship.position, [
            np.log(ship._halite / 100 + 1), np.log(me._halite / 100 + 1),
            (nearest_shipyard[0] - ship.position[0]) / 5,
            (nearest_shipyard[1] - ship.position[1]) / 5
        ], self.part_size)
        step = self.states[np.argmax(self.model.predict(inpt))]
        return step


class ShipCollectOrDepositNet(BaseNet):
    def __init__(self, part_size):
        super().__init__(part_size)
        self.actions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, None]

    def init_model(self):
        inputs = keras.Input(shape=(self.part_size * self.part_size * 3 + 4,), name="digits")
        x = keras.layers.Dense(16, activation="tanh", name="dense_1", kernel_initializer='random_normal',
                               bias_initializer='random_normal')(inputs)
        x = keras.layers.Dense(16, activation="relu", name="dense_4", kernel_initializer='random_normal',
                               bias_initializer='random_normal')(x)
        outputs = keras.layers.Dense(5, activation="softmax", name="predictions", kernel_initializer='random_normal',
                                     bias_initializer='random_normal')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def action(self, ship, board):
        me = board.current_player

        nearest_shipyard = get_nearest_shipyard(ship, board)

        inpt = get_data(board, ship.position, [
            np.log(ship._halite / 100 + 1), np.log(me._halite / 100 + 1),
            (nearest_shipyard[0] - ship.position[0]) / 5,
            (nearest_shipyard[1] - ship.position[1]) / 5
        ], self.part_size)
        step = self.actions[np.argmax(self.model.predict(inpt))]
        return step