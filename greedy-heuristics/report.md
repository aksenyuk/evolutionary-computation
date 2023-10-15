# Report

Team members:

- Sofya Aksenyuk, 150284
- Uladzimir Ivashka, 150281

## Problem Description

Given a set of nodes, each characterized by their (x, y) coordinates in a plane and an associated cost, the challenge is to select exactly 50% of these nodes and form a Hamiltonian cycle. 

The goal is to minimize the sum of the total length of the path plus the total cost of the selected nodes. 

Distances between nodes are computed as Euclidean distances and rounded to the nearest integer. 

## Methodologies

### Random Solution 
A straightforward method where a solution is formed by randomly selecting nodes until a Hamiltonian cycle is created

### Nearest Neighbor 
This algorithm starts from an initial node and sequentially selects the nearest unvisited node until all nodes are included in the path

### Greedy Cycle 
This heuristic method initiates a cycle and iteratively selects a node to insert into the cycle, minimizing the increase in total distance at each step

## Source code

Link: [Source Code](https://github.com/aksenyuk/evolutionary-computation/blob/main/greedy-heuristics/greedy_heuristics.ipynb)

## Pseudocode

### 1. Random Search

	Function RANDOM_SEARCH(distance_matrix, current_node_index=None):
	    to_visit <- SET of all node indices
	    IF current_node_index is None:
		current_node_index <- RANDOM choice from to_visit
	    ENDIF
	    
	    current_node <- current_node_index
	    solution <- LIST containing current_node
	    total_cost <- 0
	    
	    REMOVE current_node from to_visit
	    
	    WHILE to_visit is not empty:
		next_node_index <- RANDOM choice from to_visit
		ADD distance from current_node to next_node to total_cost
		ADD next_node to solution
		REMOVE next_node from to_visit
		current_node <- next_node
	    ENDWHILE
	    
	    ADD distance from last node to first node to total_cost
	    ADD first node to solution
	    
	    RETURN solution, total_cost
	END Function


### Nearest Neighbor

	Function NEAREST_NEIGHBOR(distance_matrix, current_node_index=None):
	    to_visit <- SET of all node indices
	    IF current_node_index is None:
	        current_node_index <- RANDOM choice from to_visit
	    ENDIF
	
	    current_node <- current_node_index
	    solution <- LIST containing current_node
	    total_cost <- 0
	
	    REMOVE current_node from to_visit
	
	    WHILE to_visit is not empty:
	        closest_neighbor <- MINIMUM distance neighbor from current_node in to_visit
	        closest_neighbor_distance <- distance from current_node to closest_neighbor
	
	        ADD closest_neighbor_distance to total_cost
	        ADD closest_neighbor to solution
	        REMOVE closest_neighbor from to_visit
	        current_node <- closest_neighbor
	    ENDWHILE
	
	    ADD distance from last node to first node to total_cost
	    ADD first node to solution
	
	    RETURN solution, total_cost
	END Function


### Greedy Cycle

	Function GREEDY_CYCLE(distance_matrix, current_node_index=None):
	    to_visit <- SET of all node indices
	    IF current_node_index is None:
	        current_node_index <- RANDOM choice from to_visit
	    ENDIF
	
	    current_node <- current_node_index
	    solution <- LIST containing current_node
	    total_cost <- 0
	
	    REMOVE current_node from to_visit
	
	    WHILE to_visit is not empty:
	        closest_neighbor <- NULL
	        closest_neighbor_distance <- INFINITY
	        closest_neighbor_position <- NULL
	
	        FOR EACH neighbor IN to_visit:
	            IF solution length is 1:
	                neighbor_distance <- SUM of distances between current_node and neighbor and neighbor and current_node
	                candidate_position <- 1
	            ELSE:
	                distances <- LIST of modified distances considering insertion between all nodes in solution
	                neighbor_distance, candidate_position <- MINIMUM distance and corresponding position in distances
	            ENDIF
	
	            IF neighbor_distance < closest_neighbor_distance:
	                closest_neighbor <- neighbor
	                closest_neighbor_distance <- neighbor_distance
	                closest_neighbor_position <- candidate_position
	            ENDIF
	        ENDFOR
	
	        ADD closest_neighbor_distance to total_cost
	        INSERT closest_neighbor at closest_neighbor_position in solution
	        REMOVE closest_neighbor from to_visit
	    ENDWHILE
	
	    ADD distance from last node to first node to total_cost
	    ADD first node to solution
	
	    RETURN solution, total_cost
	END Function


# Computational Experiments

## Results

### TSPA
| Algorithm         | Min     | Avg       | Max     |
|-------------------|---------|-----------|---------|
| random_search     | 504381.0| 530265.29 | 566168.0|
| nearest_neighbor  | 266571.0| 268736.18 | 270825.0|
| greedy_cycle      | 233906.0| 235950.42 | 238314.0|

### TSPB
| Algorithm         | Min     | Avg       | Max     |
|-------------------|---------|-----------|---------|
| random_search     | 498947.0| 531610.39 | 560287.0|
| nearest_neighbor  | 260275.0| 263994.80 | 266447.0|
| greedy_cycle      | 226482.0| 228284.68 | 230288.0|

### TSPC
| Algorithm         | Min     | Avg       | Max     |
|-------------------|---------|-----------|---------|
| random_search     | 393988.0| 428608.69 | 456807.0|
| nearest_neighbor  | 156765.0| 159934.69 | 163734.0|
| greedy_cycle      | 134596.0| 135505.10 | 136691.0|

### TSPD
| Algorithm         | Min     | Avg       | Max     |
|-------------------|---------|-----------|---------|
| random_search     | 405125.0| 436056.91 | 464179.0|
| nearest_neighbor  | 153543.0| 156398.39 | 159569.0|
| greedy_cycle      | 130391.0| 131439.06 | 132972.0|

## Best Solutions Plots

See plots: [Plots](https://github.com/aksenyuk/evolutionary-computation/edit/main/greedy-heuristics/plots/)

<div>
	<img src="./plots/TSPA.png" height="750"/>
	<img src="./plots/TSPB.png" height="750"/>
</div>
	
<div>
	<img src="./plots/TSPC.png" height="750"/>
	<img src="./plots/TSPD.png" height="750"/>
</div>


# Conclusions

### Performance

- Greedy Cycle generally outperformed both Nearest Neighbor and Random Search in minimizing costs across all instances, demonstrating its superior heuristic

- Nearest Neighbor also notably surpassed the random approach in delivering lower-cost solutions

### Stability of Solutions

- Random Search, though computationally light, yielded high variability and less optimal solutions

- Both greedy methods, especially Greedy Cycle, produced more stable and reliable outcomes with lower variability in costs

