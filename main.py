from kaggle_environments.envs.halite.helpers import *
from base import *


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

        return me.next_actions
