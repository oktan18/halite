from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

import numpy as np
from collections import Counter
import time

from extra import agent as extra_agent


def eval_env(env):
    res = [x[0] for x in env.steps[-1][0]['observation']['players']]
    r = []
    for i in range(len(res)):
        score = res[i]
        max_other = max([res[j] for j in range(len(res)) if i != j])
        r += [score-max_other]
    return r


def eval_model(model, board_size=20):
    environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
    environment.run([model, extra_agent, extra_agent, extra_agent])
    return eval_env(environment)


def fit(nets):
    for net in nets:
        res = eval_model(net.model())
        score = res[0]
        net.score = score
    return nets


def selection(nets, p=0.8):
    selected = sorted(nets, key=lambda x: x.score, reverse=True)
    return selected[:int(p*len(selected))]


def birth(net1, net2):
    gen = max(net1.gen, net2.gen) + 1

    child = net1.part_size
    child.gen = gen
    randoms = [np.random.random(w.shape) for w in net1.w]
    child.w = [w1*r + w2*(1-r) for w1, w2, r in zip(net1.w, net2.w, randoms)]
    return child


def mutate(nets):
    childs = []
    for _ in range(len(nets)//2):
        net1, net2 = np.random.choice(nets, 2)
        child = birth(net1, net2)
        childs += [child, ]
    nets += childs
    return nets


def evolution():
    population = [5 for _ in range(20)]
    for gen in range(20):
        print('start population', gen)
        print('population size is ', len(population))
        start_time = time.time()
        population = fit(population)
        print('population', gen, 'fited', (time.time() - start_time)/60)
        p = 0.8 if len(population) < 40 else 0.3
        population = selection(population, p)
        print(f'best at step {gen}: {population[0].score} from generation {population[0].gen}')
        print(f'worst at step {gen}: {population[-1].score} from generation {population[-1].gen}')
        population[0].save_w(f'data/populations/pop{gen}.json')
        print('gen distribution:', Counter([net.gen for net in population]))
        population = mutate(population)