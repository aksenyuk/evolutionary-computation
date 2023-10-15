# Report

## Problem Description

Given a set of nodes, each characterized by their (x, y) coordinates in a plane and an associated cost, the challenge is to select exactly 50% of these nodes and form a Hamiltonian cycle. The goal is to minimize the sum of the total length of the path plus the total cost of the selected nodes. Distances between nodes are computed as Euclidean distances and rounded to the nearest integer. 

## Methodologies

### Random Solution 
A straightforward method where a solution is formed by randomly selecting nodes until a Hamiltonian cycle is created.

### Nearest Neighbor 
This algorithm starts from an initial node and sequentially selects the nearest unvisited node until all nodes are included in the path.

### Greedy Cycle 
This heuristic method initiates a cycle and iteratively selects a node to insert into the cycle, minimizing the increase in total distance at each step.

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

Function nearest_neighbor(distance_matrix: 2D Array of Integers, current_node_index: Optional Integer) -> Tuple(List of Integers, Integer):
    If current_node_index is None:
        current_node_index = random_element_from(set(range(size_of(distance_matrix))))

    to_visit = set(range(size_of(distance_matrix)))

    current_node = current_node_index
    solution = [current_node]
    total_cost = 0

    to_visit.remove(current_node_index)

    While to_visit is not empty:
        closest_neighbor = None
        closest_neighbor_distance = Infinity

        For each neighbor in to_visit:
            If distance_matrix[current_node][neighbor] < closest_neighbor_distance:
                closest_neighbor_distance = distance_matrix[current_node][neighbor]
                closest_neighbor = neighbor

        total_cost += closest_neighbor_distance
        solution.append(closest_neighbor)

        to_visit.remove(closest_neighbor)

        current_node = closest_neighbor

    solution.append(solution[0])
    total_cost += distance_matrix[solution[-2]][solution[0]]

    Return (solution, total_cost)
    
    
### Greedy Cycle

Function GREEDY_CYCLE(distance_matrix, current_node_index=None):
Function greedy_cycle(distance_matrix: 2D Array of Integers, current_node_index: Optional Integer) -> Tuple(List of Integers, Integer):

    to_visit = set(range(size_of(distance_matrix)))

    If current_node_index is None:
        current_node_index = random_element_from(set(range(size_of(distance_matrix))))

    current_node = current_node_index
    solution = [current_node]
    total_cost = 0
    
    to_visit.remove(current_node_index)
    
    While to_visit is not empty:
        closest_neighbor = None
        closest_neighbor_distance = Infinity
        closest_neighbor_position = None
        
        For neighbor in to_visit:
            If size_of(solution) == 1:
                neighbor_distance = distance_matrix[current_node][neighbor] + distance_matrix[neighbor][current_node]
                candidate_position = 1
            Else:
                distances = [
                    distance_matrix[solution[i-1]][neighbor] + distance_matrix[neighbor][solution[i]] - distance_matrix[solution[i-1]][solution[i]] 
                    For i in range(1, size_of(solution))
                ]
                neighbor_distance, candidate_position = min([(dist, pos) For pos, dist in enumerate(distances, start=1)])
            
            If neighbor_distance < closest_neighbor_distance:
                closest_neighbor = neighbor
                closest_neighbor_distance = neighbor_distance
                closest_neighbor_position = candidate_position
        
        total_cost += closest_neighbor_distance
        solution.insert_at_position(closest_neighbor_position, closest_neighbor)
        to_visit.remove(closest_neighbor)
    
    solution.append(solution[0])
    total_cost += distance_matrix[solution[-2]][solution[0]]
    
    Return (solution, total_cost)


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


# Conclusions

	- Efficiency of Methods: The Greedy Cycle method persistently shows the smallest average and minimal total costs across all test cases, indicating a tendency to find more optimal solutions in comparison to the Random Search and Nearest Neighbor methods.

    	- Reliability: The Nearest Neighbor and Greedy Cycle methods consistently perform better than the Random Search method, which, as anticipated, produces the highest costs due to its uninformed search strategy.

    	- Variability: The Random Search method exhibits higher variability in costs across runs compared to other methods, reflecting its stochastic nature.

    	- Visualization: 2D visualizations of the solutions, which are crucial for understanding the spatial distribution and quality of solutions, have not been presented and should be developed to enhance the analysis and comparison of different methods and solutions.

