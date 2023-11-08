# Report

Team members:

- Sofya Aksenyuk, 150284
- Uladzimir Ivashka, 150281

## Problem Description

Given a set of nodes, each characterized by their (x, y) coordinates in a plane and an associated cost, the challenge is to select exactly 50% of these nodes and form a Hamiltonian cycle. 

The goal is to minimize the sum of the total length of the path plus the total cost of the selected nodes. 

Distances between nodes are computed as Euclidean distances and rounded to the nearest integer. 

## Methodologies

### Greedy Local Search

This algorithm repeatedly makes random selections of moves (intra-route or inter-route) to explore and improve a solution. 

It aims to diversify the search and potentially find better solutions by introducing randomness into the order of exploration.

### Steepest Local Search

This algorithm systematically examines all possible moves within the neighborhood, both intra-route and inter-route, and selects the move that results in the best improvement in the objective function value. 

It aims to find the absolute best move at each step.

## Source code

Link: [Source Code](https://github.com/aksenyuk/evolutionary-computation/blob/main/local-search/local-search-fixed.ipynb)

<div style="page-break-after: always"></div>

## Pseudocode

### Greedy Local Search

    Function GreedyLocalSearch(DistanceMatrix, Costs):
        CurrentSolution = (generate initial solution using RandomSearch/GreedyHeuristic)
        Improved = True
        
        While Improved:
            NeighborsInter = (generate all possible inter-node swaps)
            NeighborsIntra = (generate all possible intra-nodes/edges swaps)
            Neighbors = NeighborsInter + NeighborsIntra
            Neighbors = Shuffle(Neighbors)
            Improved = False
            
            For each Neighbor in Neighbors:
                If (Neighbor is node swap):
                    Delta = CalculateNodeSwapDelta(CurrentSolution, DistanceMatrix, Costs)
                If (Neighbor is edge swap):
                    Delta = CalculateEdgeSwapDelta(CurrentSolution, DistanceMatrix, Costs)
                If Delta > 0:
                    CurrentSolution = Neighbor
                    Improved = True
                    Break
                    
        Return CurrentSolution


<div style="page-break-after: always"></div>

### Steepest Local Search

    Function SteepestLocalSearch(DistanceMatrix, Costs):
        CurrentSolution = (generate initial solution using RandomSearch/GreedyHeuristic)
        Improved = True
        
        While Improved:
            NeighborsInter = (generate all possible inter-node swaps)
            NeighborsIntra = (generate all possible intra-nodes/edges swaps)
            Neighbors = NeighborsInter + NeighborsIntra
            Improved = False
            
            BestNeighbor = CurrentSolution
            BestDelta = 0
            
            For each Neighbor in Neighbors:
                If (Neighbor is node swap):
                    Delta = CalculateNodeSwapDelta(CurrentSolution, DistanceMatrix, Costs)
                If (Neighbor is edge swap):
                    Delta = CalculateEdgeSwapDelta(CurrentSolution, DistanceMatrix, Costs)
                If Delta > BestDelta:
                    BestNeighbor = Neighbor
                    BestDelta = Delta
            
            If BestNeighbor != CurrentSolution:
                CurrentSolution = BestNeighbor
                Improved = True
                    
        Return CurrentSolution

<div style="page-break-after: always"></div>

# Computational Experiments

## Results

### Table of Cost

<img src="plots/costs.png"/>

### Table of Time

<img src="plots/times.png"/>

## Best Solutions Plots

See plots: [Plots](https://github.com/aksenyuk/evolutionary-computation/tree/main/local-search/plots/)

<div style="page-break-after: always"></div>

<img src="plots/Greedy-edges-GreedyHeuristic_upd.png"/>

<div style="page-break-after: always"></div>

<img src="plots/Greedy-edges-Random_upd.png"/>
	
<div style="page-break-after: always"></div>

<img src="plots/Greedy-nodes-GreedyHeuristic_upd.png"/>

<div style="page-break-after: always"></div>

<img src="plots/Greedy-nodes-Random_upd.png"/>

<div style="page-break-after: always"></div>

<img src="plots/Steepest-edges-GreedyHeuristic_upd.png"/>

<div style="page-break-after: always"></div>

<img src="plots/Steepest-edges-Random_upd.png"/>

<div style="page-break-after: always"></div>

<img src="plots/Steepest-nodes-GreedyHeuristic_upd.png"/>

<div style="page-break-after: always"></div>

<img src="plots/Steepest-nodes-Random_upd.png"/>

<div style="page-break-after: always"></div>

# Conclusions

## Starting Solution

### Cost

When looking at the average cost, starting with a greedy heuristic tends to lead to better outcomes than starting with a random solution. This is consistent across all TSP instances. 

The range (min-max) for the greedy heuristic starting solutions is also generally tighter, indicating more consistent performance.

### Time

The greedy heuristic starting solution significantly reduces the search time across all instances. The average time for greedy heuristic starts is often less than a second, with minimal variation, whereas random starts lead to higher average times and a wider range. 

This suggests that the greedy heuristic is much more efficient in finding a solution quickly.

## Nodes Swaps vs Edges Swaps

### Cost

Swapping edges seems to be more effective than swapping nodes, as seen in the lower average costs for Greedy-edges and Steepest-edges strategies across all TSP instances. 

This is particularly evident when comparing them in algorithms with Random starting solution, where edge swapping consistently yields better costs.

### Time

The time complexity for edges swap is generally higher than for nodes swap, but the time difference between edges and nodes swaps is minimal.

## Greedy vs Steepest Search

### Cost

The greedy search and steepest search methods perform comparably in terms of cost. 

It was expected that steepest search would be generaly better than greedy, but that not happend. The steepest search methods have wider ranges in results, indicating more variability in their performance.

### Time

Greedy searches are much faster than steepest searches. The average times for greedy searches are consistently lower than those for steepest searches, with narrower ranges. 

This indicates that greedy searches are not only faster but also more consistent in their time performance.
