def compute_node_swap_delta(solution, distance_matrix, swap, intra_route=True, outer_solution_costs=None):
    """
    Compute the delta (change in objective function) for a node swap and return the new solution.
    
    Parameters:
    - solution (list): A list of nodes representing the current solution.
    - distance_matrix (list of lists): Matrix representing distances between nodes.
    - swap (tuple): A tuple representing a node swap (node1, node2).
    - intra_route (bool): Flag to indicate whether to compute for intra-route (True) or inter-route (False) swaps.
    - outer_solution_costs (list): List of costs for nodes outside the solution (used for inter-route swaps).
    
    Returns:
    - delta (float): Change in the objective function value due to the swap.
    - new_solution (list): Solution after performing the swap.
    """
    node1, node2 = swap
    
    if intra_route:
        # Indices of the nodes in the solution
        idx_node1 = solution.index(node1)
        idx_node2 = solution.index(node2)
        
        # Calculate the change in objective function
        # We need to consider the edges adjacent to the swapped nodes as well
        old_distance = distance_matrix[solution[idx_node1-1]][node1] + distance_matrix[node1][solution[(idx_node1+1)%len(solution)]] \
                     + distance_matrix[solution[idx_node2-1]][node2] + distance_matrix[node2][solution[(idx_node2+1)%len(solution)]]
                     
        new_distance = distance_matrix[solution[idx_node1-1]][node2] + distance_matrix[node2][solution[(idx_node1+1)%len(solution)]] \
                     + distance_matrix[solution[idx_node2-1]][node1] + distance_matrix[node1][solution[(idx_node2+1)%len(solution)]]
        
        delta = new_distance - old_distance
        
        # Create the new solution after performing the swap
        new_solution = solution.copy()
        new_solution[idx_node1], new_solution[idx_node2] = new_solution[idx_node2], new_solution[idx_node1]
    
    else:
        # For inter-route swap, we swap a node from the solution with a node outside the solution
        idx_node1 = solution.index(node1)
        
        # Calculate the change in objective function
        old_distance = distance_matrix[solution[idx_node1-1]][node1] + distance_matrix[node1][solution[(idx_node1+1)%len(solution)]]
        
        # Check if node2 is in the distance matrix (it might not be if it's outside the solution)
        if node2 < len(distance_matrix):
            new_distance = distance_matrix[solution[idx_node1-1]][node2] + distance_matrix[node2][solution[(idx_node1+1)%len(solution)]]
        else:
            new_distance = outer_solution_costs[node2]
        
        delta = new_distance - old_distance + outer_solution_costs[node2] - outer_solution_costs[node1]
        
        # Create the new solution after performing the swap
        new_solution = solution.copy()
        new_solution[idx_node1] = node2
    
    return delta, new_solution

# Test the function for intra-route swap
test_solution_node = [0, 1, 2, 3]
test_swap_intra = (1, 3)
delta_intra, new_solution_intra = compute_node_swap_delta(test_solution_node, test_distance_matrix, test_swap_intra)

# Test the function for inter-route swap
test_swap_inter = (1, 4)
test_outer_costs = [10, 15, 20, 25, 5]
delta_inter, new_solution_inter = compute_node_swap_delta(test_solution_node, test_distance_matrix, test_swap_inter, intra_route=False, outer_solution_costs=test_outer_costs)

delta_intra, new_solution_intra, delta_inter, new_solution_inter
