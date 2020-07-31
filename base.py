from kaggle_environments.envs.halite.helpers import *

import numpy as np

DIRECTIONS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]


class Base:
    def __init__(self):
        self.weights = []

    def mutate(self):
        child = self.__class__()
        return child


class BaseShipStateManager(Base):
    def state(self, ship, board):

        me = board.current_player

        if len(me.shipyards) == 0:
            return "CONVERT"

        elif ship.halite <= 500:  # If cargo is too low, collect halite
            return "COLLECT"

        elif ship.halite > 50:  # If cargo gets very big, deposit halite
            return "DEPOSIT"

        elif ship.halite == -1000:  # If cargo gets very big, deposit halite
            return "ATTACK"


class BaseShipyardManager(Base):
    def action(self, shipyard, board):
        me = board.current_player

        # If there are no ships, use first shipyard to spawn a ship.
        if len(me.ships) == 0:
            return ShipyardAction.SPAWN
        else:
            return None


class BaseCollectManager(Base):
    def action(self, ship, board):
        if ship.cell.halite < 100:
            neighbors = [ship.cell.north.halite, ship.cell.east.halite,
                         ship.cell.south.halite, ship.cell.west.halite]
            best = max(range(len(neighbors)), key=neighbors.__getitem__)
            return DIRECTIONS[best]
        else:
            return None


class BaseConvertManager(Base):
    def action(self, ship, board):
        return ShipAction.CONVERT


class BaseDepositManager(Base):
    @staticmethod
    def nearest_shipyard(ship, board):
        def distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_shipyards(board):
            me = board.current_player
            return [shipyard.position for shipyard in me.shipyards]

        ship_point = ship.position
        shipyards = get_shipyards(board)
        nearest_shipyard = shipyards[np.argmin([distance(ship_point, s) for s in shipyards])]
        return nearest_shipyard

    def action(self, ship, board):
        fromX, fromY = ship.position

        toX, toY = self.nearest_shipyard(ship, board)

        act = None
        if fromY < toY:
            act = ShipAction.NORTH
        elif fromY > toY:
            act = ShipAction.SOUTH
        if fromX < toX:
            act = ShipAction.EAST
        if fromX > toX:
            act = ShipAction.WEST

        return act