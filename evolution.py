from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

import numpy as np
from collections import Counter
import time

from extra import agent as extra_agent

from base import *
from main import HaliteManager
from nets import *


def eval_model(model, env):
    env.run([model, extra_agent, extra_agent, extra_agent])
    scores = [x[0] for x in env.steps[-1][0]['observation']['players']]
    return scores[0]


def fit(nets):
    for net in nets:
        environment = make("halite", configuration={"size": 20, "startingHalite": 1000})
        env = deepcopy(environment)
        score = eval_model(net.agent, env)
        net.score = score
    return nets


def selection(nets, p):
    selected = sorted(nets, key=lambda x: x.score, reverse=True)
    return selected[:int(p*len(selected))]


def mutate(nets):
    childs = []
    scores = np.array([net.score for net in nets])

    if max(scores) == min(scores):
        p = np.ones(len(scores))
    else:
        p = (scores - min(scores)) / (max(scores) - min(scores))
    p = p / sum(p)
    for _ in range(2*len(nets)):
        net = np.random.choice(nets, p=p)
        child = net.mutate()
        childs += [child, ]

    nets += childs
    return nets


def evolution():
    population = []
    for _ in range(100):
        shipyard_manager = BaseShipyardManager()
        collect_manager = BaseCollectManager()
        convert_manager = BaseConvertManager()
        deposit_manager = BaseDepositManager()

        ship_state_manager = BaseShipStateManager()

        net_ship_state_manager = ShipStateNet(5)
        net_deposit_manager = ShipCollectOrDepositNet(5)
        net_collect_manager = ShipCollectOrDepositNet(5)

        net = HaliteManager(
            shipyard_manager=shipyard_manager,
            ship_state_manager=ship_state_manager,
            collect_manager=collect_manager,
            convert_manager=convert_manager,
            deposit_manager=net_deposit_manager,
        )
        population += [net, ]

    for gen in range(200):
        if len(population) > 1:
            print('start population', gen)
            print('population size is ', len(population))
            start_time = time.time()
            population = fit(population)
            print('population', gen, 'fited', (time.time() - start_time) / 60)
            p = 0.334
            population = selection(population, p)
            print(f'best at step {gen}: {population[0].score} from generation {population[0].generation}')
            print(f'scores of winner: {population[0].scores}')
            print(f'worst at step {gen}: {population[-1].score} from generation {population[-1].generation}')
            population[0].save_w(f'data/populations/pop{gen}.json')
            print('gen distribution:', Counter([net.generation for net in population]))
            population = mutate(population)


if __name__ == '__main__':
    evolution()
