import argparse

# import random
import time

# import warnings
# from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def get_distance_matrix(df):
    coords = df[["x", "y"]].to_numpy()

    distance_matrix = np.round(squareform(pdist(coords, "euclidean")))
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def random_search(distance_matrix):
    n = len(distance_matrix)

    solution = list(range(n))
    np.random.shuffle(solution)

    return solution[: (n // 2)]


def get_total_cost(solution, distance_matrix, costs):
    assert len(solution) * 2 == len(distance_matrix)
    total_cost = 0

    for i in range(len(solution) - 1):
        total_cost += (
            distance_matrix[solution[i], solution[i + 1]] + costs[solution[i + 1]]
        )

    total_cost += distance_matrix[solution[-1], solution[0]] + costs[solution[0]]

    return total_cost


class Ant:
    def __init__(self, start_node):
        self.tour = [start_node]
        self.total_cost = 0

    def visit_node(self, node, cost):
        self.tour.append(node)
        self.total_cost += cost


class AntColonyOptimization:
    def __init__(
        self,
        distance_matrix,
        costs,
        end_time,
        n_ants=50,
        n_best=5,
        # n_iterations=100,
        decay=0.3,
        alpha=1,
        beta=3,
    ):
        self.distance_matrix = distance_matrix
        self.costs = costs
        self.pheromone_matrix = np.ones(self.distance_matrix.shape) / len(
            distance_matrix
        )
        self.n_ants = n_ants
        self.n_best = n_best
        self.end_time = end_time
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def _apply_pheromone_decay(self):
        self.pheromone_matrix *= 1 - self.decay

    def _update_pheromones(self, ants):
        for ant in ants:
            for i in range(len(ant.tour) - 1):
                x, y = ant.tour[i], ant.tour[i + 1]
                self.pheromone_matrix[x, y] += 1.0 / self.distance_matrix[x, y]

    def _select_next_node(self, current_node, taboo_list):
        heuristic = 1 / (self.distance_matrix[current_node] + self.costs)
        heuristic[taboo_list] = 0
        attractiveness = np.power(
            self.pheromone_matrix[current_node], self.alpha
        ) * np.power(heuristic, self.beta)

        total = np.sum(attractiveness)
        if total == 0:
            return np.random.choice(
                [
                    node
                    for node in range(len(self.distance_matrix))
                    if node not in taboo_list
                ]
            )
        probabilities = attractiveness / total
        return np.random.choice(range(len(self.distance_matrix)), p=probabilities)

    def _construct_solution(self):
        ants = [
            Ant(start_node=np.random.randint(len(self.distance_matrix)))
            for _ in range(self.n_ants)
        ]
        for _ in range(len(self.distance_matrix) // 2 - 1):
            for ant in ants:
                current_node = ant.tour[-1]
                taboo_list = ant.tour
                next_node = self._select_next_node(current_node, taboo_list)
                ant.visit_node(
                    next_node,
                    self.distance_matrix[current_node, next_node]
                    + self.costs[next_node],
                )
        return ants

    def run(self):
        start = time.time()
        best_cost = float("inf")
        best_solution = None

        while time.time() - start < self.end_time:
            ants = self._construct_solution()
            self._apply_pheromone_decay()

            sorted_ants = sorted(ants, key=lambda ant: ant.total_cost)
            for ant in sorted_ants[: self.n_best]:
                for i in range(len(ant.tour) - 1):
                    x, y = ant.tour[i], ant.tour[i + 1]
                    self.pheromone_matrix[x, y] += 1.0 / ant.total_cost

            if sorted_ants[0].total_cost < best_cost:
                best_cost = sorted_ants[0].total_cost
                best_solution = sorted_ants[0].tour

        return best_cost, best_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance")
    args = parser.parse_args()

    instance = args.instance

    end_times = {"TSPA": 1264, "TSPB": 1310, "TSPC": 1267, "TSPD": 1269}
    # end_times = {"TSPA": 1800, "TSPB": 1800, "TSPC": 1800, "TSPD": 1800}

    times = []
    costs = []
    # counters = []
    solutions = []

    for _ in range(7):
        start = time.perf_counter()

        df = pd.read_csv(f"./data/{instance}.csv", sep=";", names=["x", "y", "cost"])
        node_costs = df.cost.to_numpy()
        distance_matrix = get_distance_matrix(df)

        solver = AntColonyOptimization(distance_matrix, node_costs, end_times[instance])
        total_cost, solution = solver.run()
        end = time.perf_counter()
        total_time = end - start

        times.append(total_time)
        costs.append(total_cost)
        # counters.append(counter)
        solutions.append(solution)

    with open("results/results5.txt", "a+") as file:
        text_to_append = f"{instance} / {np.mean(costs)} ({np.min(costs)} - {np.max(costs)}) / {round(np.mean(times), 3)} ({round(np.min(times), 3)} - {round(np.max(times), 3)}) / {solutions[costs.index(min(costs))]}\n"  # noqa: E501

        file.write(text_to_append)


if __name__ == "__main__":
    main()
