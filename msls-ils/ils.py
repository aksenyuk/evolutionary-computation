import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from time import perf_counter
from itertools import combinations, product
import copy
import functools
import warnings
import os
import time
import random

warnings.filterwarnings("ignore")


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


def ils(instance, end_time):
    start = time.time()

    df = pd.read_csv(f"./data/{instance}.csv", sep=";", names=["x", "y", "cost"])
    costs = df.cost.to_numpy()
    distance_matrix = get_distance_matrix(df)

    solution = random_search(distance_matrix)
    best_total_cost = get_total_cost(solution, distance_matrix, costs)

    counter = 0
    while True:
        solution_perturbed = solution[:]
        solution_perturbed = perturb(solution_perturbed)
        new_solution = steepest_local_search(solution_perturbed, distance_matrix, costs)
        counter += 1
        new_total_cost = get_total_cost(new_solution, distance_matrix, costs)

        if new_total_cost < best_total_cost:
            best_total_cost = new_total_cost
            solution = new_solution[:]

        if time.time() - start >= end_time:
            return best_total_cost, solution, counter


def main():
    instances = ("TSPA", "TSPB", "TSPC", "TSPD")
    # end_times = (1264, 1310, 1267, 1269)
    end_times = (5, 5, 5, 5)

    for instance, end_time in zip(instances, end_times):
        start = perf_counter()
        total_cost, solution, counter = ils(instance, end_time)
        end = perf_counter()

        total_time = end - start

        with open("./results/results_ils.txt", "a+") as file:
            text_to_append = (
                f"{instance} - {total_cost} - {total_time} - {counter} - {solution}\n"
            )

            file.write(text_to_append)


if __name__ == "__main__":
    main()
