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



# Greedy Cycle

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
