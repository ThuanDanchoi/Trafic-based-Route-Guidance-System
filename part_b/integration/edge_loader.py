"""
Edge list loader for traffic graph.
"""

import csv
import os

def load_edge_list_csv(csv_path=None):
    """
    Load the edge list from a CSV file and return it as a dictionary.

    Each row in the file should contain:
        - from: Source node ID
        - to: Destination node ID
        - distance_km: Distance between nodes in kilometers

    Args:
        csv_path (str, optional): Path to the edge list CSV file. 
                                  Defaults to 'data/edge_list.csv' in the current directory.

    Returns:
        dict: A dictionary where keys are (from_node, to_node) tuples and values are distance_km.
    """
    edges = {}

    # Default path: integration/data/edge_list.csv
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'data', 'edge_list.csv')

    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                from_node = int(row['from'])
                to_node = int(row['to'])
                distance_km = float(row['distance_km'])

                edges[(from_node, to_node)] = distance_km

        print(f"Loaded {len(edges)} edges from {csv_path}")
    except Exception as e:
        print(f"Error reading file {csv_path}: {str(e)}")

    return edges
