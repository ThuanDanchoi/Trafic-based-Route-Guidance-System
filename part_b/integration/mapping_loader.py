"""
Utility functions for loading SCATS ID mappings and intersection coordinates.
"""

import csv
import os

def load_scats_mapping(csv_path=None):
    """
    Load mapping between SCATS IDs and intersection node IDs.

    For simulated data, this may not be required.
    If no CSV path is provided, attempts to read from 'data/scats_mapping.csv'.
    Returns a dictionary {scats_id: intersection_id}.

    Args:
        csv_path (str, optional): Path to SCATS mapping CSV file.

    Returns:
        dict or None: Mapping dictionary or None if file not found or error occurs.
    """
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'data', 'scats_mapping.csv')

        if not os.path.exists(csv_path):
            return None  # or return {}, depending on usage in pipeline

    mapping = {}
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scats_id = int(row['scats_id'])
                intersection_id = int(row['intersection_id'])
                mapping[scats_id] = intersection_id
        print(f"Loaded {len(mapping)} SCATS mappings from {csv_path}")
    except Exception as e:
        print(f"Error reading mapping file: {str(e)}")
        return None

    return mapping

def load_coordinates(csv_path=None):
    """
    Load intersection coordinates.

    Reads from 'data/fake_coords.csv' unless a custom path is provided.
    Returns a dictionary {intersection_id: (x, y)} representing node locations.

    Args:
        csv_path (str, optional): Path to CSV file containing coordinates.

    Returns:
        dict: Dictionary mapping node IDs to (x, y) coordinates.
    """
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'data', 'fake_coords.csv')

    coords = {}
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = int(row['intersection_id'])
                x = float(row['x_coord'])
                y = float(row['y_coord'])
                coords[node_id] = (x, y)
        print(f"Loaded {len(coords)} intersection coordinates from {csv_path}")
    except Exception as e:
        print(f"Error reading coordinate file: {str(e)}")

    return coords
