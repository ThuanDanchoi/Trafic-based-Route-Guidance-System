"""
Streamlit-based GUI for the Traffic-Based Route Guidance System (TBRGS).

This application enables users to:
- Visualize a road network graph for the Boroondara area
- Predict traffic volume at intersections using LSTM, GRU, and BiLSTM models
- Perform route finding using algorithms like A*, BFS, DFS, GBFS, CUS1, CUS2
- View node-specific traffic details and visualize search paths
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import datetime
import networkx as nx
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# Add integration directory to path
integration_path = os.path.join(os.path.dirname(__file__), 'integration')
sys.path.append(integration_path)

# Import modules from integration
from pipeline import run_pipeline
from mapping_loader import load_coordinates
from edge_loader import load_edge_list_csv

# If traffic_predictor module can be imported, use it
try:
    from traffic_predictor import TrafficPredictor
    has_predictor = True
except ImportError:
    # Mock class if module is unavailable
    class TrafficPredictor:
        def __init__(self, model_type='lstm'):
            pass

        def predict_for_time(self, datetime_obj, site_id, model_type='lstm'):
            import random
            return random.randint(50, 250)

    has_predictor = False

# Page configuration
st.set_page_config(
    page_title="🚦 TBRGS - Traffic Prediction & Route Guidance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box-success {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.5);
    }
    .info-box-warning {
        background-color: rgba(255, 165, 0, 0.1);
        border: 1px solid rgba(255, 165, 0, 0.5);
    }
    .info-box-error {
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.5);
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .route-step {
        font-size: 1rem;
        margin: 0.3rem 0;
    }
    /* Make default selectbox more attractive */
    div[data-baseweb="select"] > div {
        border-color: #FF4B4B !important;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown('<h1 class="main-header">🚗 Traffic-Based Route Guidance System (TBRGS)</h1>', unsafe_allow_html=True)
st.markdown("#### Traffic volume prediction and optimal routing for Boroondara area")


# === Load data ===
@st.cache_data
def load_all_data():
    # Paths to data files in the data directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    integration_path = os.path.join(base_path, 'integration')
    data_path = os.path.join(integration_path, 'data')

    coords_path = os.path.join(data_path, 'fake_coords.csv')
    edge_path = os.path.join(data_path, 'edge_list.csv')
    mapping_path = os.path.join(data_path, 'scats_mapping.csv')

    # Load data
    coords = load_coordinates(coords_path)
    edges = load_edge_list_csv(edge_path)

    # Load mapping
    try:
        if os.path.exists(mapping_path):
            mapping_df = pd.read_csv(mapping_path)
            mapping = dict(zip(mapping_df['scats_id'], mapping_df['intersection_id']))
        else:
            # Simulated mapping (intersection_id == scats_id)
            mapping = {node_id: node_id for node_id in coords.keys()}
    except Exception as e:
        st.error(f"Error reading mapping file: {str(e)}")
        mapping = {node_id: node_id for node_id in coords.keys()}

    # Create NetworkX graph for visualization
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for node_id, (x, y) in coords.items():
        G.add_node(node_id, pos=(x, y))

    for (from_node, to_node), distance in edges.items():
        G.add_edge(from_node, to_node, weight=distance)

    # Check if graph is connected
    if len(G.nodes) > 0:
        is_connected = nx.is_strongly_connected(G)
    else:
        is_connected = False

    road_names = {}
    try:
        # Try to read road name information from fake_coords.csv
        coords_df = pd.read_csv(coords_path)
        if 'roads' in coords_df.columns:
            for _, row in coords_df.iterrows():
                if isinstance(row['roads'], str) and row['roads']:
                    road_names[row['intersection_id']] = row['roads']
    except Exception:
        pass

    return mapping, coords, edges, G, is_connected, road_names


# Load data
mapping, coords, edge_data, graph, is_graph_connected, road_names = load_all_data()


# === Utility functions ===
def display_node_info(node_id):
    """Display detailed information about a node"""
    if node_id not in coords:
        return st.error(f"Information for node {node_id} not found")

    x, y = coords[node_id]

    # Find adjacent nodes
    neighbors_out = [(to, edge_data[(node_id, to)])
                     for (from_node, to) in edge_data.keys()
                     if from_node == node_id]
    neighbors_in = [(from_node, edge_data[(from_node, node_id)])
                    for (from_node, to) in edge_data.keys()
                    if to == node_id]

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Node Details")
        st.write(f"🔹 **Node ID:** {node_id}")

        if node_id in road_names:
            st.write(f"🔹 **Road Name:** {road_names[node_id]}")

        st.write(f"🔹 **Coordinates:** ({x:.4f}, {y:.4f})")
        st.write(f"🔹 **Outgoing Connections:** {len(neighbors_out)}")
        st.write(f"🔹 **Incoming Connections:** {len(neighbors_in)}")

    with col2:
        # Display traffic prediction if available
        st.write("#### Traffic Prediction")

        # Current or selected time
        selected_time = st.selectbox("⏱️ Select time:",
                                     ["Current", "Morning Peak", "Evening Peak", "Off-Peak"])

        now = datetime.datetime.now()
        if selected_time == "Morning Peak":
            prediction_time = now.replace(hour=8, minute=0)
        elif selected_time == "Evening Peak":
            prediction_time = now.replace(hour=17, minute=30)
        elif selected_time == "Off-Peak":
            prediction_time = now.replace(hour=14, minute=0)
        else:
            prediction_time = now

        # Predict with 3 models
        try:
            if has_predictor:
                predictors = {
                    'lstm': TrafficPredictor(model_type='lstm'),
                    'gru': TrafficPredictor(model_type='gru'),
                    'bilstm': TrafficPredictor(model_type='bilstm')
                }

                st.write(f"⏰ Prediction for: {prediction_time.strftime('%H:%M')}")

                prediction_results = {}
                for model_name, predictor in predictors.items():
                    vol = predictor.predict_for_time(prediction_time, node_id, model_name)
                    prediction_results[model_name] = int(vol)

                # Display results
                for model_name, vol in prediction_results.items():
                    model_emoji = "🧠" if model_name == "lstm" else "⚙️" if model_name == "gru" else "🔄"

                    # Classify traffic volume
                    if vol < 100:
                        traffic_status = "light 🟢"
                        color = "green"
                    elif vol < 200:
                        traffic_status = "moderate 🟡"
                        color = "orange"
                    else:
                        traffic_status = "heavy 🔴"
                        color = "red"

                    st.markdown(
                        f"{model_emoji} **{model_name.upper()}:** "
                        f"<span style='color:{color}'>{vol} vehicles/hour</span> "
                        f"({traffic_status})",
                        unsafe_allow_html=True
                    )
            else:
                st.info("⚠️ Prediction module unavailable. Displaying sample data.")
                models = ['LSTM', 'GRU', 'BiLSTM']
                for model in models:
                    vol = np.random.randint(50, 300)
                    if vol < 100:
                        traffic_status = "light 🟢"
                        color = "green"
                    elif vol < 200:
                        traffic_status = "moderate 🟡"
                        color = "orange"
                    else:
                        traffic_status = "heavy 🔴"
                        color = "red"

                    st.markdown(
                        f"🧠 **{model}:** "
                        f"<span style='color:{color}'>{vol} vehicles/hour</span> "
                        f"({traffic_status})",
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"Error during traffic prediction: {str(e)}")

    # Display connections
    st.write("#### Connections")
    col1, col2 = st.columns(2)

    with col1:
        st.write("##### Outgoing Connections")
        if neighbors_out:
            for to_node, distance in sorted(neighbors_out, key=lambda x: x[1]):
                road_info = f" ({road_names.get(to_node, 'N/A')})" if to_node in road_names else ""
                st.markdown(f"→ **Node {to_node}**{road_info}: {distance:.2f} km")
        else:
            st.info("No outgoing connections")

    with col2:
        st.write("##### Incoming Connections")
        if neighbors_in:
            for from_node, distance in sorted(neighbors_in, key=lambda x: x[1]):
                road_info = f" ({road_names.get(from_node, 'N/A')})" if from_node in road_names else ""
                st.markdown(f"← **Node {from_node}**{road_info}: {distance:.2f} km")
        else:
            st.info("No incoming connections")


def visualize_graph(graph, highlight_path=None):
    """Display graph and route (if available)"""

    if not graph.nodes:
        return st.error("No graph data to visualize")

    st.write("#### Graph Visualization")

    # Get node coordinates for drawing
    pos = nx.get_node_attributes(graph, 'pos')
    if not pos:
        # Create layout if no coordinates
        pos = nx.spring_layout(graph, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        width=0.5,
        alpha=0.5,
        arrows=True,
        arrowsize=8
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=50,
        node_color='skyblue',
        alpha=0.8
    )

    # Highlight path if available
    if highlight_path and len(highlight_path) > 1:
        path_edges = [(highlight_path[i], highlight_path[i + 1]) for i in range(len(highlight_path) - 1)]
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=path_edges,
            width=2.5,
            alpha=1,
            edge_color='red',
            arrows=True,
            arrowsize=15
        )

        # Highlight nodes on the path
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=highlight_path,
            node_size=100,
            node_color='red',
            alpha=1
        )

    # Add labels for important nodes (to avoid too many labels)
    if highlight_path:
        labels = {node: str(node) for node in highlight_path}
    else:
        # If no path, only show some important nodes
        degree = dict(graph.degree())
        important_nodes = [n for n, d in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]]
        labels = {node: str(node) for node in important_nodes}

    nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_color='darkblue')

    # Add title
    if highlight_path:
        plt.title(f"Traffic graph with marked route ({len(highlight_path)} nodes)")
    else:
        plt.title(f"Traffic graph ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")

    plt.axis('off')
    st.pyplot(fig)


def is_reachable(from_node, to_node, edge_dict):
    """Check if there is a path from from_node to to_node"""
    visited = set()
    stack = [from_node]
    while stack:
        current = stack.pop(0)  # Use BFS to find the path
        if current == to_node:
            return True
        if current in visited:
            continue
        visited.add(current)
        neighbors = [v for (u, v) in edge_dict.keys() if u == current]
        stack.extend([n for n in neighbors if n not in visited])
    return False


# === Sidebar: User input ===
st.sidebar.header("🔧 Input Settings")

# Select origin and destination
origin_options = sorted(list(coords.keys()))
destination_options = sorted(list(coords.keys()))

if road_names:
    # Display road names with IDs if available
    origin_options_with_names = [
        f"{node_id} - {road_names.get(node_id, 'Unknown')}"
        if node_id in road_names else str(node_id)
        for node_id in origin_options
    ]

    destination_options_with_names = [
        f"{node_id} - {road_names.get(node_id, 'Unknown')}"
        if node_id in road_names else str(node_id)
        for node_id in destination_options
    ]

    origin_selection = st.sidebar.selectbox(
        "🛫 Origin (Intersection ID)",
        range(len(origin_options)),
        format_func=lambda i: origin_options_with_names[i]
    )
    origin = origin_options[origin_selection]

    destination_selection = st.sidebar.selectbox(
        "🛬 Destination (Intersection ID)",
        range(len(destination_options)),
        format_func=lambda i: destination_options_with_names[i]
    )
    destination = destination_options[destination_selection]
else:
    # If no road names, display IDs only
    origin = st.sidebar.selectbox("🛫 Origin (Intersection ID)", origin_options)
    destination = st.sidebar.selectbox("🛬 Destination (Intersection ID)", destination_options)

# Model and algorithm
model_choice = st.sidebar.selectbox("🧠 ML Model", ['lstm', 'gru', 'bilstm'])
method_choice = st.sidebar.selectbox("🔍 Search Algorithm", ['AS', 'BFS', 'DFS', 'GBFS', 'CUS1', 'CUS2'])

# Display custom parameters
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Advanced Settings")

show_advanced = st.sidebar.checkbox("Show advanced parameters")
if show_advanced:
    congestion_factor = st.sidebar.slider(
        "Congestion factor",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher factor means greater impact of traffic volume on travel time"
    )

    intersection_delay = st.sidebar.slider(
        "Intersection delay (seconds)",
        min_value=0,
        max_value=60,
        value=30,
        step=5,
        help="Delay time when passing through each signalized intersection"
    )
else:
    congestion_factor = 1.0
    intersection_delay = 30

# === Debug Sidebar ===
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Graph Information**")
st.sidebar.markdown(f"📄 Total edges: `{len(edge_data)}`")
st.sidebar.markdown(f"🏙️ Total intersections: `{len(coords)}`")

connected_from_origin = [to for (frm, to) in edge_data.keys() if frm == origin]
st.sidebar.markdown(f"🔎 Node `{origin}` can reach `{len(connected_from_origin)}` nodes")

if not connected_from_origin:
    st.sidebar.warning("⚠️ Origin node has no outgoing edges!")
elif len(connected_from_origin) <= 10:
    st.sidebar.write("→", connected_from_origin)
else:
    st.sidebar.write("→", connected_from_origin[:10], "...")

st.sidebar.markdown("---")
# Node detail view functionality
st.sidebar.subheader("🔍 View node details")
node_to_inspect = st.sidebar.number_input(
    "Enter node ID to view details:",
    min_value=min(coords.keys()) if coords else 0,
    max_value=max(coords.keys()) if coords else 1000,
    value=origin
)

inspect_button = st.sidebar.button("🔍 View node details")

# === Main content area ===
tabs = st.tabs(["🧭 Route Finding", "🔍 Node Information", "📊 Data Visualization"])

with tabs[0]:
    if not is_graph_connected:
        st.warning("⚠️ Graph is not strongly connected. Some points may not be reachable from others.")

    # Check connection when points are selected
    if origin == destination:
        st.warning("⚠️ Origin and destination are the same. Please choose different points.")
    elif not is_reachable(origin, destination, edge_data):
        st.error("❌ The two points are not connected in the graph. Please choose a different pair of points.")
    else:
        # Display route finding button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"#### 🚗 Find route from `{origin}` to `{destination}`")

            # Display road names if available
            if road_names:
                origin_name = road_names.get(origin, "Unknown")
                dest_name = road_names.get(destination, "Unknown")
                st.markdown(f"**From:** {origin_name} → **To:** {dest_name}")

        with col2:
            run_button = st.button("🚀 Run Route Search", use_container_width=True)

        # Handle when route finding button is pressed
        if run_button:
            with st.spinner("🔄 Predicting and finding route..."):
                start_time = time.time()

                result = run_pipeline(
                    mapping_data=mapping,
                    coord_data=coords,
                    edge_distance_data=edge_data,
                    origin=origin,
                    destinations=[destination],
                    model=model_choice,
                    method=method_choice
                )

                execution_time = time.time() - start_time

                st.success(f"🎉 Route found in {execution_time:.2f} seconds!")

                col1, col2 = st.columns([3, 2])

                with col1:
                    st.subheader("📋 Route Information:")
                    st.markdown(f"- **Algorithm:** `{method_choice}`")
                    st.markdown(f"- **ML Model:** `{model_choice.upper()}`")
                    st.markdown(f"- **Nodes processed:** `{result['nodes_created']}`")

                    if result['goal'] is not None:
                        estimated_time = result["goal"]

                        # Convert to hours:minutes if needed
                        if estimated_time >= 60:
                            hours = int(estimated_time // 60)
                            minutes = int(estimated_time % 60)
                            time_str = f"{hours} hours {minutes} minutes"
                        else:
                            time_str = f"{round(estimated_time, 1)} minutes"

                        st.markdown(f"⏱️ **Estimated travel time:** `{time_str}`")
                    else:
                        st.error("❌ Could not find a route from the selected origin to the destination.")

                    st.subheader("🗺️ Route:")

                    # Display route as a numbered list
                    if result['path']:
                        path = result['path']
                        for i, node in enumerate(path):
                            road_info = f" ({road_names.get(node, '')})" if node in road_names else ""
                            if i == 0:
                                st.markdown(f"1. **Start at Node {node}**{road_info}")
                            elif i == len(path) - 1:
                                st.markdown(f"{i + 1}. **End at Node {node}**{road_info}")
                            else:
                                st.markdown(f"{i + 1}. Node {node}{road_info}")
                    else:
                        st.warning("No route data available")

                with col2:
                    # Visualize route on graph
                    if result['path']:
                        visualize_graph(graph, result['path'])
                    else:
                        visualize_graph(graph)

with tabs[1]:
    # Display node details when button is pressed or node info tab is selected
    if inspect_button or tabs[1].id:
        display_node_info(node_to_inspect)
    else:
        st.info("👈 Enter a node ID and press 'View node details' in the sidebar to see detailed information")

with tabs[2]:
    st.subheader("📊 Traffic Data Visualization")

    # Display overview graph
    st.markdown("#### 🗺️ Traffic Graph Map")
    visualize_graph(graph)

    # Graph statistics
    st.markdown("#### 📊 Graph Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Intersections", len(coords))

    with col2:
        st.metric("Total Roads", len(edge_data))

    with col3:
        # Calculate graph density (edges / nodes)
        if len(coords) > 0:
            density = len(edge_data) / len(coords)
            st.metric("Graph Density", f"{density:.2f}")
        else:
            st.metric("Graph Density", "N/A")

    # Analysis and statistics
    if graph.nodes:
        st.markdown("#### 📊 Node Analysis")

        # Calculate degree distribution
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]

        # Create histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(in_degrees, bins=max(5, min(20, max(in_degrees))), alpha=0.7, color='skyblue')
        ax1.set_title('In-degree Distribution')
        ax1.set_xlabel('In-degree')
        ax1.set_ylabel('Frequency')

        ax2.hist(out_degrees, bins=max(5, min(20, max(out_degrees))), alpha=0.7, color='lightgreen')
        ax2.set_title('Out-degree Distribution')
        ax2.set_xlabel('Out-degree')
        ax2.set_ylabel('Frequency')

        plt.tight_layout()
        st.pyplot(fig)

        # Display most important nodes (with highest connectivity)
        st.markdown("#### 🌟 Top 10 Most Important Intersections")

        # Calculate node importance as sum of in-degree and out-degree
        node_importance = {}
        for node in graph.nodes():
            node_importance[node] = graph.in_degree(node) + graph.out_degree(node)

        # Sort by importance in descending order
        important_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        # Create dataframe for display
        important_df = pd.DataFrame(important_nodes, columns=['Node ID', 'Connections'])

        # Add road names if available
        if road_names:
            important_df['Road Name'] = important_df['Node ID'].apply(
                lambda x: road_names.get(x, "Unknown")
            )

        st.dataframe(important_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "### 🚦 Traffic-Based Route Guidance System (TBRGS) - COS30019 Introduction to AI"
)
st.markdown(
    "Traffic volume prediction and optimal routing system for Boroondara area, Melbourne"
)