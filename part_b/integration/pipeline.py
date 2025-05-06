"""
Integration pipeline for traffic prediction and route planning.
"""

from datetime import datetime
import os
import sys

# Add the root directory of the project to path to find part_a
current_dir = os.path.dirname(os.path.abspath(__file__))  # integration directory
parent_dir = os.path.dirname(current_dir)  # part_b directory
project_dir = os.path.dirname(parent_dir)  # project root directory

# Append paths to sys.path
sys.path.append(parent_dir)       # Add part_b to path
sys.path.append(project_dir)      # Add project root to path

from integration.traffic_predictor import TrafficPredictor
from integration.traffic_graph_builder import build_edge_distances_from_coords

try:
    # Try importing from part_a
    from part_a.graph import Graph
    from part_a.algorithms.astar import search as astar_search
    from part_a.algorithms.bfs import search as bfs_search
    from part_a.algorithms.dfs import search as dfs_search
    from part_a.algorithms.gbfs import search as gbfs_search
    from part_a.algorithms.cus1 import search as cus1_search
    from part_a.algorithms.cus2 import search as cus2_search
except ModuleNotFoundError:
    # If part_a is not found, use absolute path
    print("Cannot find module part_a. Trying to import from absolute path...")
    # Assume part_a is at the same level as part_b
    part_a_path = os.path.join(os.path.dirname(parent_dir), 'part_a')
    sys.path.append(part_a_path)

    # Try importing again
    try:
        from graph import Graph
        from algorithms.astar import search as astar_search
        from algorithms.bfs import search as bfs_search
        from algorithms.dfs import search as dfs_search
        from algorithms.gbfs import search as gbfs_search
        from algorithms.cus1 import search as cus1_search
        from algorithms.cus2 import search as cus2_search

        print("Successfully imported from absolute path!")
    except ModuleNotFoundError as e:
        print(f"Still cannot find module: {e}")
        print(f"Tried paths: {sys.path}")
        raise


def run_pipeline(mapping_data, coord_data, edge_distance_data,
                 origin, destinations, model='lstm', method='AS'):
    """
       Execute the route planning pipeline

       Args:
           mapping_data (dict): Maps SCATS IDs to node IDs
           coord_data (dict): Maps node IDs to (x, y) coordinates
           edge_distance_data (dict): Maps (from_node, to_node) to distance in kilometers
           origin (int): Starting node ID
           destinations (list): List of destination node IDs
           model (str): Type of traffic prediction model to use ('lstm', 'gru', 'bilstm')
           method (str): Search method to use ('AS', 'BFS', 'DFS', 'GBFS', 'CUS1', 'CUS2')

       Returns:
           dict: A dictionary containing the goal node, number of nodes created,
                 the final path, and the full graph used.
       """

    predictor = TrafficPredictor(model_type=model)
    now = datetime.now()

    # Predict traffic volume at each node
    predicted_volumes = {}
    for scats_id, node_id in mapping_data.items():
        vol = predictor.predict_for_time(now, scats_id, model)
        predicted_volumes[node_id] = vol

    # Create graph with travel time cost on each edge
    nodes = coord_data
    edges = {}

    for (from_node, to_node), distance_km in edge_distance_data.items():
        volume_at_B = predicted_volumes.get(to_node, 5)

        # Clamp volume within reasonable bounds
        volume_at_B = max(10, min(volume_at_B, 300))

        # Calculate travel time (minutes)
        # Assume average speed is 30km/h
        base_time_min = (distance_km / 30) * 60
        congestion_factor = 1 + volume_at_B / 1000
        travel_time_min = (distance_km / 30) * 60 * (1 + volume_at_B / 2000)

        edges[(from_node, to_node)] = travel_time_min

    graph = Graph(nodes, edges, origin, destinations)

    # Call the selected search algorithm
    if method == 'AS':
        goal, nodes_created, path = astar_search(graph)
    elif method == 'BFS':
        goal, nodes_created, path = bfs_search(graph)
    elif method == 'DFS':
        goal, nodes_created, path = dfs_search(graph)
    elif method == 'GBFS':
        goal, nodes_created, path = gbfs_search(graph)
    elif method == 'CUS1':
        goal, nodes_created, path = cus1_search(graph)
    elif method == 'CUS2':
        goal, nodes_created, path = cus2_search(graph)
    else:
        raise ValueError(f"Method {method} is not supported in pipeline.py")

    return {
        "goal": goal,
        "nodes_created": nodes_created,
        "path": path,
        "graph": graph
    }
