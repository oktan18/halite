import json
from base import *


class HaliteManager:
    def __init__(
            self,
            shipyard_manager: BaseShipyardManager,
            ship_state_manager: BaseShipStateManager,
            collect_manager: BaseCollectManager,
            convert_manager: BaseConvertManager,
            deposit_manager: BaseDepositManager
    ):
        self.ship_state_manager = ship_state_manager
        self.shipyard_manager = shipyard_manager

        self.collect_manager = collect_manager
        self.convert_manager = convert_manager
        self.deposit_manager = deposit_manager

        self.state_dict = {
            "COLLECT": self.collect,
            "CONVERT": self.convert,
            "DEPOSIT": self.deposit,
        }

        self._scores = list()

        self.generation = 0

    @property
    def score(self):
        if len(self._scores) > 0:
            # return np.mean(self._scores)
            return self._scores[-1]
        else:
            return 0

    @property
    def scores(self):
        return self._scores

    @score.setter
    def score(self, s):
        self._scores += [s, ]

    def deposit(self, ship, board):
        return self.deposit_manager.action(ship, board)

    def collect(self, ship, board=None):
        return self.collect_manager.action(ship, board)

    def convert(self, ship=None, board=None):
        return self.convert_manager.action(ship, board)

    def agent(self, observation, configuration):
        board = Board(observation, configuration)
        me = board.current_player
        actions_to_apply = []

        for shipyard in me.shipyards:
            shipyard.next_action = self.shipyard_manager.action(shipyard, board)
            actions_to_apply += (shipyard, shipyard.next_action)
            board.next()

        for ship in me.ships:
            state = self.ship_state_manager.state(ship, board)

            ship.next_action = self.state_dict[state](ship, board)

            actions_to_apply += (ship, ship.next_action)
            board.next()
            #
            # if isinstance(self.ship_state_manager, ShipStateNet):
            #     print('state', state)
            #     print('act', ship.next_action)
        for item, act in actions_to_apply:
            item.next_action = act
        return me.next_actions

    def mutate(self):
        child_shipyard_manager = self.shipyard_manager.mutate()
        child_ship_state_manager = self.ship_state_manager.mutate()
        child_collect_manager = self.collect_manager.mutate()
        child_convert_manager = self.convert_manager.mutate()
        child_deposit_manager = self.deposit_manager.mutate()

        child = self.__class__(
            child_shipyard_manager,
            child_ship_state_manager,
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
        self.collect_manager.weights = wx['collect_manager']
        self.convert_manager.weights = wx['convert_manager']
        self.deposit_manager.weights = wx['deposit_manager']

