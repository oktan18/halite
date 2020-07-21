from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

import numpy as np
from collections import Counter
import time

from extra import agent as extra_agent

from base import *
from main import HaliteManager
from nets import *


def eval_env(env):
    res = [x[0] for x in env.steps[-1][0]['observation']['players']]
    r = []
    for i in range(len(res)):
        score = res[i]
        max_other = max([res[j] for j in range(len(res)) if i != j])
        r += [score-max_other]
    # print(res)
    return res


def eval_model(model, board_size=20):
    environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
    environment.run([model, extra_agent, extra_agent, extra_agent])
    return eval_env(environment)


def fit(nets):
    for net in nets:
        res = eval_model(lambda x, y: net.agent(x, y))
        score = res[0]
        net.score = score
    return nets


def selection(nets, p=0.8):
    selected = sorted(nets, key=lambda x: x.score, reverse=True)
    return selected[:int(p*len(selected))]


def mutate(nets):
    childs = []
    scores = np.array([net.score for net in nets])

    if max(scores) == min(scores):
        p = np.ones(len(scores))
    else:
        p = scores - min(scores) / (max(scores) - min(scores)) + 1
    p = p / sum(p)
    for _ in range(len(nets)):
        net = np.random.choice(nets, p=p)
        child = net.mutate()
        childs += [child, ]

    nets += childs
    return nets


def evolution():
    population = []
    for _ in range(20):
        shipyard_manager = BaseShipyardManager()
        attack_manager = BaseAttackManager()
        collect_manager = BaseCollectManager()
        convert_manager = BaseConvertManager()
        deposit_manager = BaseDepositManager()

        net_ship_state_manager = ShipStateNet(5)

        net = HaliteManager(
            shipyard_manager=shipyard_manager,
            ship_state_manager=net_ship_state_manager,
            attack_manager=attack_manager,
            collect_manager=collect_manager,
            convert_manager=convert_manager,
            deposit_manager=deposit_manager,
        )
        population += [net, ]

    for gen in range(50):
        print('start population', gen)
        print('population size is ', len(population))
        start_time = time.time()
        population = fit(population)
        print('population', gen, 'fited', (time.time() - start_time)/60)
        p = 0.6 if len(population) < 30 else 0.3
        population = selection(population, p)
        print(f'best at step {gen}: {population[0].score} from generation {population[0].generation}')
        print(f'worst at step {gen}: {population[-1].score} from generation {population[-1].generation}')
        population[0].save_w(f'data/populations/pop{gen}.json')
        print('gen distribution:', Counter([net.generation for net in population]))
        population = mutate(population)


if __name__ == '__main__':
    evolution()
