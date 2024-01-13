# import copy
# import functools
import argparse
import random
import time
import warnings
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# import ast


warnings.filterwarnings("ignore")

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors


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


def compute_inter_move_delta(solution, distance_matrix, costs, idx, new_node):
    n = len(solution)
    new_solution = solution.copy()
    old_node = solution[idx]

    new = (
        costs[new_node]
        + distance_matrix[new_solution[idx - 1], new_node]
        + distance_matrix[new_node, new_solution[(idx + 1) % n]]
    )

    old = (
        costs[old_node]
        + distance_matrix[new_solution[idx - 1], old_node]
        + distance_matrix[old_node, new_solution[(idx + 1) % n]]
    )

    delta = new - old
    new_solution[idx] = new_node

    return new_solution, delta


def compute_intra_move_delta(solution, distance_matrix, indices, backward=False):
    ## without roll/shift to initial form
    n = len(solution)
    i, j = indices

    if i >= j:
        raise Exception("Wrong indices, i >= j")
    if j >= n:
        raise Exception("Wrong indices, j >= n")

    if backward:
        if (i == 0 and j in (n - 1, n - 2)) or (j == n - 1 and i in (0, 1)):
            return solution, 0
        new = (
            distance_matrix[solution[i], solution[(j + 1) % n]]
            + distance_matrix[solution[j], solution[i - 1]]
        )
        old = (
            distance_matrix[solution[i - 1], solution[i]]
            + distance_matrix[solution[j], solution[(j + 1) % n]]
        )
    else:
        if j - i in (1, 2):
            return solution, 0
        new = (
            distance_matrix[solution[i], solution[j - 1]]
            + distance_matrix[solution[i + 1], solution[j]]
        )
        old = (
            distance_matrix[solution[i], solution[i + 1]]
            + distance_matrix[solution[j - 1], solution[j]]
        )

    delta = new - old

    if backward:
        new_solution = (
            solution[j + 1 :][::-1] + solution[i : j + 1] + solution[:i][::-1]
        )
    else:
        new_solution = solution[: i + 1] + solution[i + 1 : j][::-1] + solution[j:]

    return new_solution, delta


def steepest_local_search(solution, distance_matrix, costs):
    solution = solution[:]
    n, N = len(solution), len(distance_matrix)
    solution_set = set(solution)
    outer_nodes_set = set(range(N)) - solution_set

    while True:
        best_delta, best_solution = 0, None
        inter_move_flag, inter_move_outer_node, inter_move_inner_node_idx = (
            False,
            None,
            None,
        )

        # inter
        for outer_node, inner_node_idx in product(outer_nodes_set, range(n)):
            new_solution, delta = compute_inter_move_delta(
                solution, distance_matrix, costs, inner_node_idx, outer_node
            )
            if delta < best_delta:
                best_delta = delta
                best_solution = new_solution[:]
                inter_move_flag = True
                inter_move_outer_node, inter_move_inner_node_idx = (
                    outer_node,
                    inner_node_idx,
                )

        # intra
        for i, j in combinations(range(n), 2):
            # forward
            new_solution, delta = compute_intra_move_delta(
                solution, distance_matrix, (i, j), False
            )
            if delta < best_delta:
                best_delta = delta
                best_solution = new_solution[:]
                inter_move_flag = False
            # backward
            new_solution, delta = compute_intra_move_delta(
                solution, distance_matrix, (i, j), True
            )
            if delta < best_delta:
                best_delta = delta
                best_solution = new_solution[:]
                inter_move_flag = False

        if best_solution is not None:
            if inter_move_flag:
                solution_set.add(inter_move_outer_node)
                solution_set.remove(solution[inter_move_inner_node_idx])
                outer_nodes_set.remove(inter_move_outer_node)
                outer_nodes_set.add(solution[inter_move_inner_node_idx])
            solution = best_solution[:]
            continue
        return solution


def greedy_2_regret_weighted(
    distance_matrix, partial_solution, costs, target_size, regret_weight=0.5
):
    num_nodes = len(distance_matrix)
    to_visit = set(range(num_nodes)) - set(partial_solution)

    solution = partial_solution[:]

    while len(solution) < target_size:
        max_weighted_sum = float("-inf")
        best_node = None
        best_insertion_point = None

        for node in to_visit:
            insertion_costs = []
            for i in range(len(solution) - 1):
                cost = (
                    distance_matrix[solution[i]][node]
                    + distance_matrix[node][solution[i + 1]]
                    - distance_matrix[solution[i]][solution[i + 1]]
                    + costs[node]
                )
                insertion_costs.append((cost, i))

            insertion_costs.append(
                (
                    distance_matrix[solution[-1]][node]
                    + distance_matrix[node][solution[0]]
                    - distance_matrix[solution[-1]][solution[0]]
                    + costs[node],
                    len(solution) - 1,
                )
            )

            insertion_costs.sort(key=lambda x: x[0])

            weighted_sum = 0
            if len(insertion_costs) > 1:
                regret = insertion_costs[1][0] - insertion_costs[0][0]
                objective = -insertion_costs[0][0]
                weighted_sum = regret_weight * regret + (1 - regret_weight) * objective

            if weighted_sum > max_weighted_sum:
                max_weighted_sum = weighted_sum
                best_node = node
                best_insertion_point = insertion_costs[0][1]

        solution.insert(best_insertion_point + 1, best_node)
        to_visit.remove(best_node)

    return solution


def random_insertion(solution, n=5):
    for i in range(n):
        which, where = random.randint(0, len(solution) - 1), random.randint(
            0, len(solution)
        )
        solution.insert(where, solution.pop(which))
    return solution


def double_bridge_move(solution):
    a, b, c = sorted(random.sample(range(1, len(solution) - 2), 3))
    return solution[:a] + solution[c:] + solution[b:c] + solution[a:b]


def shuffle_sub_tour(solution):
    n = len(solution)
    sub_tour_length = random.randint(int(0.05 * n), int(0.15 * n))
    start_idx = random.randint(0, n - sub_tour_length)
    end_idx = start_idx + sub_tour_length

    sub_tour = solution[start_idx:end_idx]
    random.shuffle(sub_tour)
    solution = solution[:start_idx] + sub_tour + solution[end_idx:]

    return solution


def random_jump(solution):
    n = len(solution)
    sub_tour_length = random.randint(int(0.05 * n), int(0.15 * n))
    start_idx = random.randint(0, n - sub_tour_length)
    end_idx = start_idx + sub_tour_length
    sub_tour = solution[start_idx:end_idx]

    new_solution = solution[:start_idx] + solution[end_idx:]

    insert_idx = random.randint(0, len(new_solution))
    new_solution = new_solution[:insert_idx] + sub_tour + new_solution[insert_idx:]

    return new_solution


def k_opt_move(solution, k=4):
    n = len(solution)
    edges = sorted(random.sample(range(n), k))
    edges.append(edges[0])

    new_solution = []
    for i in range(k):
        start, end = edges[i], edges[i + 1]
        if start < end:
            new_solution.extend(solution[start:end])
        else:
            new_solution.extend(solution[start:] + solution[:end])

        if i % 2 == 1:
            new_solution[-(end - start) :] = reversed(new_solution[-(end - start) :])

    return new_solution


def perturb(solution):
    perturbations = (
        random_insertion,
        double_bridge_move,
        shuffle_sub_tour,
        random_jump,
        k_opt_move,
    )
    action = random.choice(perturbations)
    solution = action(solution)
    return solution


def get_initial_population(pop_size, distance_matrix, costs):
    population = []
    for _ in range(pop_size):
        solution = random_search(distance_matrix)
        new_solution = steepest_local_search(solution, distance_matrix, costs)
        population.append(new_solution)
    return population


def operator_1(parent1, parent2):
    length = len(parent1)
    child = [None] * length

    edges1 = {
        tuple(sorted((parent1[i], parent1[(i + 1) % length]))) for i in range(length)
    }
    edges2 = {
        tuple(sorted((parent2[i], parent2[(i + 1) % length]))) for i in range(length)
    }
    common_edges = edges1.intersection(edges2)

    for i in range(length - 1):
        edge = tuple(sorted((parent1[i], parent1[i + 1])))
        if edge in common_edges:
            child[i] = parent1[i]
            child[i + 1] = parent1[i + 1]

    common_nodes = set(parent1) & set(parent2) - set(child)

    for node in common_nodes:
        for i in range(length):
            if child[i] is None:
                child[i] = node
                break

    remaining_nodes = [node for node in range(length * 2) if node not in child]
    random.shuffle(remaining_nodes)
    for i in range(length):
        if child[i] is None:
            child[i] = remaining_nodes.pop()

    return child


def operator_2(parent1, parent2, distance_matrix, costs):
    length = len(parent1)

    edges1 = {
        tuple(sorted((parent1[i], parent1[(i + 1) % length]))) for i in range(length)
    }
    edges2 = {
        tuple(sorted((parent2[i], parent2[(i + 1) % length]))) for i in range(length)
    }
    common_edges = edges1.intersection(edges2)

    modified_parent = []
    for i in range(length - 1):
        edge = tuple(sorted((parent1[i], parent1[i + 1])))
        if edge in common_edges:
            if not modified_parent or modified_parent[-1] != parent1[i]:
                modified_parent.append(parent1[i])
            modified_parent.append(parent1[i + 1])

    modified_parent = greedy_2_regret_weighted(
        distance_matrix, modified_parent, costs, length
    )

    return modified_parent


def simulated_annealing(
    initial_solution,
    distance_matrix,
    costs,
    T=10000,
    cooling_rate=0.995,
    stopping_temperature=1,
):
    current_solution = initial_solution[:]
    current_cost = get_total_cost(current_solution, distance_matrix, costs)
    best_solution = current_solution[:]
    best_cost = current_cost

    while T > stopping_temperature:
        neighbor = perturb(current_solution.copy())
        neighbor_cost = get_total_cost(neighbor, distance_matrix, costs)

        if neighbor_cost < current_cost or np.random.uniform(0, 1) < np.exp(
            (current_cost - neighbor_cost) / T
        ):
            current_solution = neighbor[:]
            current_cost = neighbor_cost

            if neighbor_cost < best_cost:
                best_solution = neighbor[:]
                best_cost = neighbor_cost

        T *= cooling_rate

    return best_solution


def evol_algo(instance, end_time):
    start = time.time()

    df = pd.read_csv(f"./data/{instance}.csv", sep=";", names=["x", "y", "cost"])
    costs = df.cost.to_numpy()
    distance_matrix = get_distance_matrix(df)

    population = get_initial_population(20, distance_matrix, costs)
    total_costs = [
        get_total_cost(solution, distance_matrix, costs) for solution in population
    ]

    counter = 0
    while time.time() - start < end_time:
        counter += 1

        if random.randint(0, 1) == 0:
            if random.randint(0, 1) == 0:
                parent1, parent2 = random.sample(population, 2)
                oper_num = random.randint(1, 2)
                if oper_num == 1:
                    child = operator_1(parent1, parent2)
                elif oper_num == 2:
                    child = operator_2(parent1, parent2, distance_matrix, costs)
            else:
                parent = random.choice(population)
                child = perturb(parent)
            child = steepest_local_search(child, distance_matrix, costs)

        else:
            child = simulated_annealing(child, distance_matrix, costs)

        child_total_cost = get_total_cost(child, distance_matrix, costs)

        if child_total_cost not in total_costs:
            max_total_cost = max(total_costs)
            if child_total_cost < max_total_cost:
                max_total_cost_idx = total_costs.index(max_total_cost)
                total_costs[max_total_cost_idx] = child_total_cost
                population[max_total_cost_idx] = child

    best_total_cost = min(total_costs)
    best_solution = population[total_costs.index(best_total_cost)]
    return best_total_cost, best_solution, counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance")
    args = parser.parse_args()

    instance = args.instance

    # end_times = {"TSPA": 1264, "TSPB": 1310, "TSPC": 1267, "TSPD": 1269}
    end_times = {"TSPA": 1500, "TSPB": 1500, "TSPC": 1500, "TSPD": 1500}

    times = []
    costs = []
    counters = []
    solutions = []

    for _ in range(20):
        start = time.perf_counter()
        total_cost, solution, counter = evol_algo(instance, end_times[instance])
        end = time.perf_counter()
        total_time = end - start

        times.append(total_time)
        costs.append(total_cost)
        counters.append(counter)
        solutions.append(solution)

    with open("results/results2.txt", "a+") as file:
        text_to_append = f"{instance} / {np.mean(costs)} ({np.min(costs)} - {np.max(costs)}) / {round(np.mean(times), 3)} ({round(np.min(times), 3)} - {round(np.max(times), 3)}) / {np.mean(counters)} ({np.min(counters)} - {np.max(counters)}) / {solutions[costs.index(min(costs))]}\n"  # noqa: E501

        file.write(text_to_append)


if __name__ == "__main__":
    main()
