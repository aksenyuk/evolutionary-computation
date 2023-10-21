# Report

Team members:

- Sofya Aksenyuk, 150284
- Uladzimir Ivashka, 150281

## Problem Description

Given a set of nodes, each characterized by their (x, y) coordinates in a plane and an associated cost, the challenge is to select exactly 50% of these nodes and form a Hamiltonian cycle. 

The goal is to minimize the sum of the total length of the path plus the total cost of the selected nodes. 

Distances between nodes are computed as Euclidean distances and rounded to the nearest integer. 

## Methodologies

### Greedy 2-Regret

This heuristic focuses on the regret value of not choosing the second-best option. For each unvisited node, it computes the difference in cost between the best edge and the second-best edge. The algorithm then selects the node with the highest regret value and adds it to the current path.

### Greedy 2-Regret Weighted

A variation of the Greedy 2-Regret heuristic where weights are considered. This can provide a more balanced consideration between the regret value and the actual distance.

## Source code

Link: [Source Code](https://github.com/aksenyuk/evolutionary-computation/blob/main/greedy-regret-heuristics/greedy_regret_heuristics.ipynb)

## Pseudocode

### Greedy 2-Regret

    FUNCTION greedy_2_regret(distance_matrix, current_node_index=None):
        Initialize to_visit with all nodes
        IF current_node_index is None:
            Select a random starting node
        Initialize solution with the starting node
        REMOVE starting node from to_visit
    
        WHILE solution is less than half of all nodes:
            FOR each node in to_visit:
                Calculate insertion costs of adding node between all pairs of consecutive nodes in solution
                Determine the regret of the best insertion position
                IF regret is greater than previous maximum:
                    Update best node and position to insert
            Insert the best node into the solution at the determined position
            Remove the best node from to_visit
    
        Calculate the total cost of the solution path
        RETURN solution, total_cost

### Greedy 2-Regret Weighted

    FUNCTION greedy_2_regret_weighted(distance_matrix, current_node_index=None, regret_weight=0.5):
        Initialize to_visit with all nodes
        IF current_node_index is None:
            Select a random starting node
        Initialize solution with the starting node
        REMOVE starting node from to_visit
    
        WHILE solution is less than half of all nodes:
            FOR each node in to_visit:
                Calculate insertion costs of adding node between all pairs of consecutive nodes in solution
                Calculate the weighted sum of regret and objective for best insertion position
                IF weighted sum is greater than previous maximum:
                    Update best node and position to insert
            Insert the best node into the solution at the determined position
            Remove the best node from to_visit
    
        Calculate the total cost of the solution path
        RETURN solution, total_cost

# Computational Experiments

## Results

<img src="https://github.com/aksenyuk/evolutionary-computation/blob/main/greedy-regret-heuristics/plots/results.png"/>

## Best Solutions Plots

See plots: [Plots](https://github.com/aksenyuk/evolutionary-computation/edit/main/greedy-regret-heuristics/plots/)

<div>
	<img src="https://github.com/aksenyuk/evolutionary-computation/blob/main/greedy-regret-heuristics/plots/TSPA.png" height="750"/>
	<img src="https://github.com/aksenyuk/evolutionary-computation/blob/main/greedy-regret-heuristics/plots/TSPB.png" height="750"/>
</div>
	
<div>
	<img src="https://github.com/aksenyuk/evolutionary-computation/blob/main/greedy-regret-heuristics/plots/TSPC.png" height="750"/>
	<img src="https://github.com/aksenyuk/evolutionary-computation/blob/main/greedy-regret-heuristics/plots/TSPD.png" height="750"/>
</div>


# Conclusions

## Performance

- Greedy 2-Regret Weighted performed closely with the Greedy Cycle, often delivering competitive or even better results in certain instances
- Greedy 2-Regret, while better than Random Search, did not perform as well as the Greedy Cycle or Nearest Neighbor

## Stability of Solutions

- Greedy 2-Regret showed a consistent performance, though not always optimal, across all instances
- Greedy 2-Regret Weighted exhibited a high degree of stability, often surpassing the Greedy Cycle in consistency
