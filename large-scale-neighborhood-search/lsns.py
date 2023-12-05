import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import time
from itertools import combinations, product
import argparse

import warnings

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


def destroy(solution, distance_matrix, costs):
    n = len(solution)
    num_to_destroy = int(np.random.normal(25, 10))
    num_to_destroy = max(min(num_to_destroy, 50), 10)

    weights = np.zeros(n)
    for i in range(n - 1):
        prev, cur, next_ = solution[i - 1], solution[i], solution[i + 1]
        weight = distance_matrix[prev][cur] + costs[cur] + distance_matrix[cur][next_]
        weights[i] = weight
    weight = distance_matrix[-2][-1] + costs[-1] + distance_matrix[-1][0]
    weights[-1] = weight
    weights /= sum(weights)

    indeces_to_destroy = set(
        np.random.choice(n, size=num_to_destroy, replace=False, p=weights)
    )

    solution = [item for i, item in enumerate(solution) if i not in indeces_to_destroy]

    return solution


def lsns(instance, end_time, use_local):
    start = time.time()

    df = pd.read_csv(f"./data/{instance}.csv", sep=";", names=["x", "y", "cost"])
    costs = df.cost.to_numpy()
    distance_matrix = get_distance_matrix(df)

    solution = random_search(distance_matrix)
    solution = steepest_local_search(solution, distance_matrix, costs)
    best_total_cost = get_total_cost(solution, distance_matrix, costs)
    counter = 0

    while time.time() - start < end_time:
        destroyed_solution = destroy(solution, distance_matrix, costs)

        new_solution = greedy_2_regret_weighted(
            distance_matrix, destroyed_solution, costs, target_size=100
        )

        if use_local:
            new_solution = steepest_local_search(new_solution, distance_matrix, costs)

        new_total_cost = get_total_cost(new_solution, distance_matrix, costs)

        if new_total_cost < best_total_cost:
            best_total_cost = new_total_cost
            solution = new_solution[:]

        counter += 1

    return best_total_cost, solution, counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ls")
    args = parser.parse_args()
    use_local = (True, False)[args.ls == "False"]

    instances = ("TSPA", "TSPB", "TSPC", "TSPD")
    end_times = (1264, 1310, 1267, 1269)
    # end_times = (10, 10, 10, 10)

    for instance, end_time in zip(instances, end_times):
        start = time.perf_counter()
        total_cost, solution, counter = lsns(instance, end_time, use_local)
        end = time.perf_counter()

        total_time = end - start

        with open(f"./results/results_lsns_{str(use_local)}.txt", "a+") as file:
            text_to_append = (
                f"{instance} - {total_cost} - {total_time} - {counter} - {solution}\n"
            )

            file.write(text_to_append)


if __name__ == "__main__":
    main()
