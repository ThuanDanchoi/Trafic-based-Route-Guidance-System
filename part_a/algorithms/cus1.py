"""
Iterative Deepening Depth-First Search (IDDFS) - Custom Algorithm 1
"""

def dfs_with_depth_limit(graph, node, path, visited, depth_limit, nodes_created):
    """
    Depth-limited DFS helper function
    """
    if depth_limit == 0:
        return None, nodes_created, []
    
    if graph.is_destination(node):
        return node, nodes_created, path
    
    visited.add(node)
    
    neighbors = graph.get_neighbors(node)
    
    for neighbor, _ in neighbors:
        if neighbor not in visited:
            new_path = path + [neighbor]
            nodes_created += 1
            goal, new_nodes_created, result_path = dfs_with_depth_limit(
                graph, neighbor, new_path, visited.copy(), depth_limit - 1, nodes_created
            )
            
            nodes_created = new_nodes_created
            
            if goal is not None:
                return goal, nodes_created, result_path
    
    return None, nodes_created, []

def search(graph):
    origin = graph.origin
    max_depth = 1000  # Depth limit to prevent infinite loops
    nodes_created = 1
    
    for depth in range(1, max_depth):
        visited = set()
        goal, new_nodes_created, path = dfs_with_depth_limit(
            graph, origin, [origin], visited, depth, nodes_created
        )
        
        nodes_created = new_nodes_created
        
        if goal is not None:
            return goal, nodes_created, path
    
    return None, nodes_created, []

