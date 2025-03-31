"""
A* Search (A*)
"""

import heapq

def search(graph):
    origin = graph.origin
    queue = []
    counter = 0  # Counter to ensure chronological ordering when ties occur
    
    # Initial node: f = g + h, where g = 0 and h = heuristic
    h = graph.heuristic(origin)
    # Priority: f-value, then node_id, then counter
    heapq.heappush(queue, (h, origin, counter, [origin], 0))
    
    g_values = {origin: 0}
    visited = set()   
    
    nodes_created = 1
    
    while queue:
        f, node, _, path, g = heapq.heappop(queue)
        
        if node in visited:
            continue
            
        visited.add(node)
        
        if graph.is_destination(node):
            return node, nodes_created, path
        
        for neighbor, cost in graph.get_neighbors(node):
            new_g = g + cost
   
            if neighbor not in g_values or new_g < g_values[neighbor]:
                g_values[neighbor] = new_g
                h = graph.heuristic(neighbor)
                f = new_g + h
                
                counter += 1
                new_path = path + [neighbor]
                # Priority by f-value, then by node_id (smaller has priority), then by counter
                heapq.heappush(queue, (f, neighbor, counter, new_path, new_g))
                nodes_created += 1
    
    return None, nodes_created, []