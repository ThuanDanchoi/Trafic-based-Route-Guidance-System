"""
Greedy Best-First Search (GBFS)  
"""

import heapq

def search(graph):
    origin = graph.origin
    queue = []
    counter = 0  # Counter to ensure chronological ordering when ties occur

    h = graph.heuristic(origin)
    # Priority: h-value, then node_id, then counter
    heapq.heappush(queue, (h, origin, counter, [origin]))

    visited = set()
    nodes_created = 1

    while queue:
        _, node, _, path = heapq.heappop(queue)

        if node in visited:
            continue
        
        visited.add(node)

        if graph.is_destination(node):
            return node, nodes_created, path
        
        neighbors = graph.get_neighbors(node)

        for neighbor, _ in neighbors:
            if neighbor not in visited:
                counter += 1
                h = graph.heuristic(neighbor)
                new_path = path + [neighbor]
                # Priority by h-value, then by node_id (smaller has priority), then by counter
                heapq.heappush(queue, (h, neighbor, counter, new_path))
                nodes_created += 1

    return None, nodes_created, []