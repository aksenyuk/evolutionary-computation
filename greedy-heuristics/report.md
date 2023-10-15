# Randome Search

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



# Nearest Neighbor

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




# Greedy Cycle

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

