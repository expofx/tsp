import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import argparse

# arguments

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--Population", help = "Population size", type = int, default = 100)
parser.add_argument("-c", "--Cities", help = "Number of cities", type = int, default = 10)
parser.add_argument("-s", "--Selection", help = "Selection size", type = int, default = 20)
parser.add_argument("-m", "--Mutation", help = "Mutation rate", type = float, default = 0.01)
parser.add_argument("-g", "--Generations", help = "Number of generations", type = int, default = 100)
args = parser.parse_args()

# assign variables

P = args.Population
C = args.Cities
S = args.Selection
M = args.Mutation
G = args.Generations

# create initial population

def path(cities):

    path = random.sample(cities, len(cities))
    return path

def population(P, cities):

    pop = []

    for i in range(P):
        pop.append(path(cities))

    return pop

# calculate path length

def distance(x1, y1, x2, y2):

    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist

def path_length(path):

    path_length = 0

    for i in range(len(path)-1):
        path_length += distance(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
    path_length += distance(path[-1][0], path[-1][1], path[0][0], path[0][1])

    return path_length

# calculate fitness

def fitness(pop):

    fit = {}

    for i in range(len(pop)):
        fit[i] = (1/path_length(pop[i]))

    return sorted(fit.items(), key=lambda x: x[1], reverse=True)

# select mating pool

def fitness_proportionate_selection(ranked_path, S):

    selection = []
    
    df = pd.DataFrame(np.array(ranked_path), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
    
    for i in range(0, S):
        selection.append(ranked_path[i][0])

    for i in range(len(ranked_path) - S):
        pick = 100*random.random()
        for i in range(len(ranked_path)):
            if pick <= df.iat[i,3]:
                selection.append(ranked_path[i][0])
                break

    return selection

def mating_pool(pop, selection):

    mating_pool = []

    for i in range(len(selection)):
        mating_pool.append(pop[selection[i]])

    return mating_pool

# breed

def ordered_crossover(parent1, parent2):

    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breed(mating_pool, S):

    children = []
    length = len(mating_pool) - S
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(S):
        children.append(mating_pool[i])
    
    for i in range(length):
        child = ordered_crossover(pool[i], pool[len(mating_pool)-i-1])
        children.append(child)

    return children

# mutate

def swap_mutation(child, M):

    for i in range(len(child)):
        if random.random() < M:
            swap_with = int(random.random() * len(child))
            child[i], child[swap_with] = child[swap_with], child[i]

    return child

def mutate(children, M):

    mutated_pop = []

    for i in range(len(children)):
        mutated_pop.append(swap_mutation(children[i], M))

    return mutated_pop

# next generation

def next_generation(pop, S, M):

    ranked_path = fitness(pop)
    selection = fitness_proportionate_selection(ranked_path, S)
    mating = mating_pool(pop, selection)
    children = breed(mating, S)
    next_gen = mutate(children, M)

    return next_gen

# plot cities

x, y = np.random.rand(C), np.random.rand(C)

fig = plt.gcf()
fig.show()
fig.canvas.draw()

plt.scatter(x, y)
plt.plot()
# plt.show()
plt.pause(0.00000001)
fig.canvas.draw()

cities = list(zip(x,y))

# run algorithm

def run_algorithm(P, S, M, G):

    pop = population(P, cities) # initial population

    progress = []
    best_path = []

    for i in range(G):
        pop = next_generation(pop, S, M)
        progress.append(1/fitness(pop)[0][1])

        best_path = pop[0]

        plt.clf()

        for i in range(len(best_path)-1):
            plt.plot([best_path[i][0], best_path[i+1][0]], [best_path[i][1], best_path[i+1][1]], 'b-')
        plt.plot([best_path[-1][0], best_path[0][0]], [best_path[-1][1], best_path[0][1]], 'b-')

        plt.scatter(x, y)
        plt.pause(0.000000001)
        fig.canvas.draw()

    # show final path

    for i in range(len(best_path)-1):
        plt.plot([best_path[i][0], best_path[i+1][0]], [best_path[i][1], best_path[i+1][1]], 'o-')
    plt.plot([best_path[-1][0], best_path[0][0]], [best_path[-1][1], best_path[0][1]], 'o-')

    plt.scatter(x, y)
    plt.show()

    # show learning curve

    plt.clf()

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    
    best_idx = fitness(pop)[0][0]
    best_path = pop[best_idx]

    print("Distance: ", 1/fitness(pop)[0][1])

run_algorithm(P, S, M, G)