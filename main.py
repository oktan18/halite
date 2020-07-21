import json

from kaggle_environments.envs.halite.helpers import *
from base import *
from nets import ShipStateNet


class HaliteManager:
    def __init__(
            self,
            shipyard_manager: BaseShipyardManager,
            ship_state_manager: BaseShipStateManager,
            attack_manager: BaseAttackManager,
            collect_manager: BaseCollectManager,
            convert_manager: BaseConvertManager,
            deposit_manager: BaseDepositManager
    ):
        self.ship_state_manager = ship_state_manager
        self.shipyard_manager = shipyard_manager

        self.collect_manager = collect_manager
        self.attack_manager = attack_manager
        self.convert_manager = convert_manager
        self.deposit_manager = deposit_manager

        self.state_dict = {
            "COLLECT": self.collect,
            "CONVERT": self.convert,
            "ATTACK": self.attack,
            "DEPOSIT": self.deposit,
        }

        self._scores = list()

        self.generation = 0

    @property
    def score(self):
        if len(self._scores) > 0:
            return np.mean(self._scores)
        else:
            return 0

    @score.setter
    def score(self, s):
        self._scores += [s, ]

    def deposit(self, ship, observation, configuration):
        return self.deposit_manager.action(ship, observation, configuration)

    def collect(self, ship, observation=None, configuration=None):
        return self.collect_manager.action(ship, observation, configuration)

    def attack(self, ship, observation=None, configuration=None):
        return self.attack_manager.action(ship, observation, configuration)

    def convert(self, ship=None, observation=None, configuration=None):
        return self.convert_manager.action(ship, observation, configuration)

    def agent(self, observation, configuration):
        board = Board(observation, configuration)
        me = board.current_player

        for shipyard in me.shipyards:
            shipyard.next_action = self.shipyard_manager.action(shipyard, observation, configuration)

        for ship in me.ships:
            state = self.ship_state_manager.state(ship, observation, configuration)

            ship.next_action = self.state_dict[state](ship, observation, configuration)
            #
            # if isinstance(self.ship_state_manager, ShipStateNet):
            #     print('state', state)
            #     print('act', ship.next_action)

        return me.next_actions

    def mutate(self):
        child_shipyard_manager = self.shipyard_manager.mutate()
        child_ship_state_manager = self.ship_state_manager.mutate()
        child_attack_manager = self.attack_manager.mutate()
        child_collect_manager = self.collect_manager.mutate()
        child_convert_manager = self.convert_manager.mutate()
        child_deposit_manager = self.deposit_manager.mutate()

        child = self.__class__(
            child_shipyard_manager,
            child_ship_state_manager,
            child_attack_manager,
            child_collect_manager,
            child_convert_manager,
            child_deposit_manager,
        )
        child.generation = self.generation + 1

        return child

    def save_w(self, filename):
        wx = {
            'shipyard_manager': [w1.tolist() for w1 in self.shipyard_manager.weights],
            'ship_state_manager': [w1.tolist() for w1 in self.ship_state_manager.weights],
            'attack_manager': [w1.tolist() for w1 in self.attack_manager.weights],
            'collect_manager': [w1.tolist() for w1 in self.collect_manager.weights],
            'convert_manager': [w1.tolist() for w1 in self.convert_manager.weights],
            'deposit_manager': [w1.tolist() for w1 in self.deposit_manager.weights]
        }
        with open(filename, 'w') as f:
            json.dump(wx, f)

    def load_w(self, filename):
        with open(filename, 'r') as f:
            wx = json.load(f)
            wx = {
                manager: [np.array(w) for w in weights] for manager, weights in wx.items()
            }

        self.shipyard_manager.weights = wx['shipyard_manager']
        self.ship_state_manager.weights = wx['ship_state_manager']
        self.attack_manager.weights = wx['attack_manager']
        self.collect_manager.weights = wx['collect_manager']
        self.convert_manager.weights = wx['convert_manager']
        self.deposit_manager.weights = wx['deposit_manager']

