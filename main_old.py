from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

import numpy as np
import keras
from collections import Counter
import time
import json

from extra import agent as extra_agent, get_data, SHIP_ACTS, getDirTo


def get_move(board, point, model, ship_halite, full_halite, part_size=5):
    a = get_data(board, point, ship_halite, full_halite, part_size)
    step = SHIP_ACTS[np.argmax(model.predict(a))]
    return step


def eval_model(model, board_size=20):
    environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
    environment.run([agent(model), extra_agent, extra_agent, extra_agent])
    return eval_env(environment)



class Net:
    def __init__(self, part_size):

        self.part_size = part_size
        self.gen = 0
        self.score = 0
        self.wins = 0
        self._model = None

        # self._acts = np.random.choice(['relu', 'tanh', 'linear'], 3)
        self.acts = ['relu', 'relu', 'relu']
        self.w = None

    def init_hyperparams(self):
        hyperparams = {
            'acts': self.acts,
            'w': self.w,
        }
        return hyperparams

    @property
    def model(self):
        if self._model is None:
            inputs = keras.Input(shape=(self.part_size * self.part_size * 5 + 2,), name="digits")
            x = keras.layers.Dense(128, activation=self.acts[0], name="dense_1", kernel_initializer='random_normal',
                                   bias_initializer='random_normal')(inputs)
            x = keras.layers.Dense(64, activation=self.acts[1], name="dense_2", kernel_initializer='random_normal',
                                   bias_initializer='random_normal')(x)
            x = keras.layers.Dense(64, activation=self.acts[2], name="dense_3", kernel_initializer='random_normal',
                                   bias_initializer='random_normal')(x)
            x = keras.layers.Dense(32, activation=self.acts[2], name="dense_4", kernel_initializer='random_normal',
                                   bias_initializer='random_normal')(x)
            x = keras.layers.Dense(32, activation=self.acts[2], name="dense_5", kernel_initializer='random_normal',
                                   bias_initializer='random_normal')(x)
            x = keras.layers.Dense(32, activation=self.acts[2], name="dense_6", kernel_initializer='random_normal',
                                   bias_initializer='random_normal')(x)
            x = keras.layers.Dense(32, activation=self.acts[2], name="dense_7", kernel_initializer='random_normal',
                                   bias_initializer='random_normal')(x)
            outputs = keras.layers.Dense(7, activation="softmax", name="predictions", kernel_initializer='random_normal',
                                         bias_initializer='random_normal')(x)
            self._model = keras.Model(inputs=inputs, outputs=outputs)

            if self.w is None:
                self.w = [x.numpy() for x in self._model.weights]
        return self._model

    def play(self, board_size, mode='train'):

        environment = make("halite", configuration={"size": board_size, "startingHalite": 1000}, debug=True)
        environment.run([agent(self.model), extra_agent, extra_agent, extra_agent])
        environment.render(mode="ipython", width=800, height=600)

    def save_w(self, filename):
        if self.w is None:
            wx = {}
        else:
            wx = [w1.tolist() for w1 in self.w]
        with open(filename, 'w') as f:
            json.dump(wx, f)

    def load_w(self, filename):
        with open(filename, 'r') as f:
            wx = json.load(f)
            wx = [np.array(w) for w in wx]

        if wx == {}:
            wx = None
        self.w = wx

    def train(self, x, y, epochs):
        self.model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        self.model.fit(x, y, epochs=epochs, verbose=False)


def agent(model):
    def agent_m(model, observation, configuration):
        size = configuration.size
        board = Board(observation, configuration)

        me = board.current_player
        if len(me.ships) == 0 and len(me.shipyards) > 0:
            me.shipyards[0].next_action = ShipyardAction.SPAWN

        # If there are no shipyards, convert first ship into shipyard.
        if len(me.shipyards) == 0 and len(me.ships) > 0:
            me.ships[0].next_action = ShipAction.CONVERT

        for ship in me.ships:
            if ship.next_action is None:
                step = get_move(board, ship.position, model, ship.halite, me.halite)
                if step == "DEPOSIT":
                    # Move towards shipyard to deposit cargo
                    direction = getDirTo(ship.position, me.shipyards[0].position, size)
                    if direction:
                        ship.next_action = direction
                    else:
                        ship.next_action = None
                else:
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