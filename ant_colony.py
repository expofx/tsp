import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

# arguments

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--Alpha", help = "Distance power", type = int, default = 1)
parser.add_argument("-b", "--Beta", help = "Pheromone power", type = int, default = 10)
parser.add_argument("-i", "--Intensity", help = "Pheromone intensity", type = int, default = 10)
parser.add_argument("-e", "--Evaporation", help = "Pheromone evaporation rate", type = float, default = 0.5)
parser.add_argument("-n", "--Number", help = "Number of ants", type = int, default = 20)
parser.add_argument("-c", "--Cities", help = "Number of cities", type = int, default = 10)
parser.add_argument("-g", "--Generations", help = "Number of generations", type = int, default = 100)
args = parser.parse_args()

# assign variables

I = args.Intensity  # aka q
E = args.Evaporation    # aka rho
N = args.Number
C = args.Cities
A = args.Alpha
B = args.Beta
G = args.Generations

# plot cities

# C = 100

x, y = np.random.rand(C), np.random.rand(C)

fig = plt.gcf()
fig.show()
fig.canvas.draw()

plt.scatter(x, y)
plt.plot()
plt.pause(0.00000001)
fig.canvas.draw()

cities = list(zip(x,y))

# calculate distance matrix

def distance(x1, y1, x2, y2):
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist

dist_matrix = np.zeros((C,C))

for i in range(C):
    for j in range(C):
        dist_matrix[i][j] = distance(cities[i][0], cities[i][1], cities[j][0], cities[j][1])

# create graph for each ant (adapted from https://github.com/ppoffice/ant-colony-tsp)

class Graph(object):
    def __init__(self, dist_matrix, C):
        self.matrix = dist_matrix
        self.rank = C
        self.pheromone = [[1/(C*C) for j in range(C)] for i in range(C)]

# create ants

class ACO(object):
    def __init__(self, N, G, A, B, E, I):
        self.Q = I
        self.rho = E
        self.beta = B
        self.alpha = A
        self.ant_count = N
        self.generations = G

    def _update_pheromone(self, graph, ants):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]
    
    def solve(self, graph):
        best_cost = float('inf')
        best_solution = []
        progress = []

        for gen in range(self.generations):
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            for ant in ants:
                for i in range(graph.rank - 1):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)

            # progress

            progress.append(best_cost)

            # plot best solution

            plt.clf()

            best_path = [cities[i] for i in best_solution]

            for i in range(len(best_path)-1):
                plt.plot([best_path[i][0], best_path[i+1][0]], [best_path[i][1], best_path[i+1][1]], 'b-')
            plt.plot([best_path[-1][0], best_path[0][0]], [best_path[-1][1], best_path[0][1]], 'b-')

            plt.scatter(x, y)
            plt.pause(0.000000001)
            fig.canvas.draw()

        return progress, best_path, best_cost

class _Ant(object):
    def __init__(self, aco, graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []
        self.pheromone_delta = []
        self.allowed = [i for i in range(graph.rank)]
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in range(graph.rank)]

        start = random.randint(0, graph.rank - 1)
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][i] ** self.colony.beta
        
        probabilities = [0 for i in range(self.graph.rank)]
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                    self.eta[self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass

        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]

# run ACO

aco = ACO(N, G, A, B, E, I)
graph = Graph(dist_matrix, C)
progress, best_path, distance = aco.solve(graph)

# plot final solution

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

print("Distance: ", distance)