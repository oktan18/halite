import keras
from kaggle_environments.envs.halite.helpers import *

from base import BaseShipyardManager
import numpy as np
from preprocess import *


class BaseNet:
    def __init__(self, part_size=5):
        self.model = None
        self.part_size = part_size
        self.init_model()
        self._weights = [x.numpy() for x in self.model.weights]
        self._scores = []

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
        child = self.__class__(self.part_size)
        w = [wi + np.random.normal(size=wi.shape)*(np.random.random(wi.shape) < 0.05) for wi in self.weights]
        child.weights = w
        return child


class ShipStateNet(BaseNet):
    def __init__(self, part_size):

        self.states = ["CONVERT", "COLLECT", "DEPOSIT", "ATTACK"]
        super().__init__(part_size)

    def init_model(self):
        inputs = keras.Input(shape=(self.part_size * self.part_size * 3 + 2,), name="digits")
        x = keras.layers.Dense(64, activation="linear", name="dense_1", kernel_initializer='random_normal',
                               bias_initializer='random_normal')(inputs)
        x = keras.layers.Dense(64, activation="linear", name="dense_2", kernel_initializer='random_normal',
                               bias_initializer='random_normal')(inputs)
        x = keras.layers.Dense(32, activation="relu", name="dense_3", kernel_initializer='random_normal',
                               bias_initializer='random_normal')(inputs)
        x = keras.layers.Dense(32, activation="relu", name="dense_4", kernel_initializer='random_normal',
                               bias_initializer='random_normal')(x)
        outputs = keras.layers.Dense(4, activation="softmax", name="predictions", kernel_initializer='random_normal',
                                     bias_initializer='random_normal')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def state(self, ship, observation, configuration):

        board = Board(observation, configuration)
        me = board.current_player
        if len(me.shipyards) == 0:
            return "CONVERT"

        inpt = get_data(board, ship.position, ship.halite, me.halite, self.part_size)
        step = self.states[np.argmax(self.model.predict(inpt))]
        return step
