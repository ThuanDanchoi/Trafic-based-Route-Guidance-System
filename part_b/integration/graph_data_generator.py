"""
Graph data generator for simulated traffic intersections.

Outputs:
    - fake_coords.csv
    - edge_list.csv
    - scats_mapping.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import random

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Move up to parent (part_b/)
parent_dir = os.path.dirname(current_dir)

# List of possible input paths
possible_paths = [
    os.path.join(current_dir, "integrated_traffic_spatial_data.csv"),
    os.path.join(current_dir, "data", "integrated_traffic_spatial_data.csv"),
    os.path.join(parent_dir, "integrated_traffic_spatial_data.csv"),
]

input_data_path = None
for path in possible_paths:
    if os.path.exists(path):
        input_data_path = path
        break

if input_data_path is None:
    print("File 'integrated_traffic_spatial_data.csv' not found!")
    print("Please place the file in one of the following locations:")
    for path in possible_paths:
        print(f"  - {path}")
    sys.exit(1)

# Output directory
output_dir = os.path.join(current_dir, "data")
os.makedirs(output_dir, exist_ok=True)

print(f"🔍 Reading data from {input_data_path}...")

try:
    # Load original CSV data
    df = pd.read_csv(input_data_path)

    # Extract unique intersections and coordinates
    coords_df = df[['intersection_id', 'x_coord', 'y_coord']].dropna().drop_duplicates()

    # Add road column if available
    if 'roads' in df.columns:
        roads_df = df[['intersection_id', 'roads']].dropna().drop_duplicates()
        coords_df = coords_df.merge(roads_df, on='intersection_id', how='left')
    else:
        coords_df['roads'] = ""

    coords_df = coords_df.astype({'intersection_id': int, 'x_coord': float, 'y_coord': float})

    print(f"Loaded {len(coords_df)} unique intersections from data")

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for _, row in coords_df.iterrows():
        G.add_node(
            row['intersection_id'],
            x=row['x_coord'],
            y=row['y_coord'],
            roads=row['roads'] if isinstance(row['roads'], str) else ""
        )

    # Prepare for edge creation
    coords = coords_df[['x_coord', 'y_coord']].values
    ids = coords_df['intersection_id'].values

    if len(coords) < 2:
        print("Not enough points to generate graph!")
        sys.exit(1)

    # Use Nearest Neighbors to find nearby nodes
    num_neighbors = min(6, len(coords))  # including itself
    nn = NearestNeighbors(n_neighbors=num_neighbors)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    edges = []
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        from_id = ids[i]
        for dist, j in zip(dists[1:], idxs[1:]):  # skip self
            to_id = ids[j]

            distance_km = round(dist / 1000, 3)  # Assume coordinates in meters

            if distance_km <= 2.0:
                G.add_edge(from_id, to_id, distance=distance_km)
                edges.append({'from': from_id, 'to': to_id, 'distance_km': distance_km})

    # Ensure graph is strongly connected
    if not nx.is_strongly_connected(G):
        components = list(nx.strongly_connected_components(G))
        print(f"Graph has {len(components)} strongly connected components")

        if len(components) > 1:
            largest_component = max(components, key=len)

            for component in components:
                if component != largest_component:
                    node_from = random.choice(list(component))
                    node_to = random.choice(list(largest_component))

                    from_coords = (G.nodes[node_from]['x'], G.nodes[node_from]['y'])
                    to_coords = (G.nodes[node_to]['x'], G.nodes[node_to]['y'])

                    dx = from_coords[0] - to_coords[0]
                    dy = from_coords[1] - to_coords[1]
                    distance_km = round(np.sqrt(dx ** 2 + dy ** 2) / 1000, 3)

                    G.add_edge(node_from, node_to, distance=distance_km)
                    G.add_edge(node_to, node_from, distance=distance_km)

                    edges.append({'from': node_from, 'to': node_to, 'distance_km': distance_km})
                    edges.append({'from': node_to, 'to': node_from, 'distance_km': distance_km})

                    print(f"  👉 Connected node {node_from} <-> {node_to} with distance {distance_km}km")

    is_connected = nx.is_strongly_connected(G)
    print(f"Final graph connectivity: {'Connected' if is_connected else 'NOT Connected'}")

    # Export node coordinates
    coords_output = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        coords_output.append({
            'intersection_id': node_id,
            'x_coord': node_data['x'],
            'y_coord': node_data['y'],
            'roads': node_data['roads']
        })

    coords_df_output = pd.DataFrame(coords_output)
    edges_df = pd.DataFrame(edges)

    coords_file = os.path.join(output_dir, "fake_coords.csv")
    edges_file = os.path.join(output_dir, "edge_list.csv")

    coords_df_output.to_csv(coords_file, index=False)
    edges_df.to_csv(edges_file, index=False)

    print(f"Generated {len(edges)} edges")
    print(f"Saved node coordinates to {coords_file}")
    print(f"Saved edge list to {edges_file}")

    # Generate test node list
    random_nodes = random.sample(list(G.nodes()), min(10, len(G.nodes())))
    print("\n🔍 Sample nodes for testing:")
    for node in random_nodes:
        print(f"  Node {node}: has {len(list(G.successors(node)))} outgoing edges")

    # Save SCATS mapping (scats_id = intersection_id)
    mapping_df = pd.DataFrame({
        'scats_id': list(G.nodes()),
        'intersection_id': list(G.nodes())
    })
    mapping_file = os.path.join(output_dir, "scats_mapping.csv")
    mapping_df.to_csv(mapping_file, index=False)
    print(f"Saved SCATS mapping to {mapping_file}")

    # Copy original data to output directory (if not already there)
    if input_data_path != os.path.join(output_dir, "integrated_traffic_spatial_data.csv"):
        import shutil
        destination = os.path.join(output_dir, "integrated_traffic_spatial_data.csv")
        shutil.copyfile(input_data_path, destination)
        print(f"Copied original data to {destination}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
