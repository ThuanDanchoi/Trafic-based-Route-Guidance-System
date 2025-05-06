# integration/traffic_graph_builder.py

import math

def euclidean_distance(coord1, coord2):
    """Tính khoảng cách Euclidean giữa hai tọa độ (x, y)"""
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def build_edge_distances_from_coords(coords_dict, edge_list):
    """
    Trả về dict {(from_node, to_node): distance_in_km}
    Args:
        coords_dict: {node_id: (x, y)}
        edge_list: list of (from_node, to_node)

    Returns:
        edge_distance_data: dict {(A, B): distance_km}
    """
    edge_distance_data = {}
    for from_node, to_node in edge_list:
        if from_node in coords_dict and to_node in coords_dict:
            dist = euclidean_distance(coords_dict[from_node], coords_dict[to_node])
            edge_distance_data[(from_node, to_node)] = dist
    return edge_distance_data
