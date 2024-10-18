import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import random

st.title("British Museum Search")
with st.expander("Instructions", expanded=False):
    st.info("""
    ### Instructions:
    1. **Choose Input Method**: You can select either **Manual Input** or **Random Graph Generation**.
    2. **Manual Input**:
    - Enter all the nodes in a comma-separated format.
    - After submitting the nodes, enter the connections for each node. If using weights, specify them in pairs (e.g., `Node1, 5, Node2, 3`).
    3. **Random Graph Generation**:
    - Specify the number of nodes and edges.
    - Click the "Generate Random Graph" button to create a random graph.
    4. **Finding a Path**:
    - Select the starting and goal nodes.
    - Click the "Find Path" button to execute a BFS search and visualize the path.
    5. **Reset**: You can reset the application using the reset button.
    """)

input_method = st.radio("Choose input method:", ["Manual Input", "Random Graph Generation"])

graph_type = st.radio("Graph Type:", ["Undirected", "Directed", "Directed + Weights", "Undirected + Weights"])

if 'graph' not in st.session_state:
    st.session_state.graph = None  
if 'nodes' not in st.session_state:
    st.session_state.nodes = []
if 'edges' not in st.session_state:
    st.session_state.edges = {}
if 'start_node' not in st.session_state:
    st.session_state.start_node = ""
if 'goal_node' not in st.session_state:
    st.session_state.goal_node = ""
if 'path' not in st.session_state:
    st.session_state.path = None
if 'input_complete' not in st.session_state:
    st.session_state.input_complete = False

if st.session_state.graph is None:
    if "Undirected" in graph_type:
        st.session_state.graph = nx.Graph()
    else:
        st.session_state.graph = nx.DiGraph()  

def visualize_graph(G, path=[]):
    plt.figure(figsize=(15, 9))
    pos = nx.spring_layout(G)
    if "Undirected" in graph_type:
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16, 
                font_color='black', edge_color='gray')  
    else:
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16, 
                font_color='black', edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20)

    if path:
        path_edges = list(zip(path, path[1:]))   
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3, arrowstyle='->', arrowsize=20)   
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='green', node_size=2000)  
        if "Weights" in graph_type:
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Graph Visualization" if not path else "Graph with British Museum Search Path")
    plt.axis('off')
    st.pyplot(plt)

def bfs_search(graph, start, goal):
    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        elif node not in visited:
            neighbors = sorted(graph.neighbors(node))
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

            visited.add(node)

    return None

if input_method == "Manual Input" and not st.session_state.input_complete:
    if not st.session_state.nodes:
        nodes_input = st.text_input("Enter all nodes (comma-separated):", key="nodes_input")
        if st.button("Submit Nodes") and nodes_input:
            st.session_state.nodes = [node.strip() for node in nodes_input.split(',')]
            st.session_state.edges = {node: [] for node in st.session_state.nodes}

if st.session_state.nodes:
    st.subheader("Enter connections for each node")
    for node in st.session_state.nodes:
        connections = st.text_input(f"Enter connections for {node}:", key=f"conn_{node}")
        if st.button(f"Submit Connections for {node}", key=f"btn_{node}") and connections:
            if "Weights" in graph_type:
                connections_list = connections.split(',')
                for i in range(0, len(connections_list), 2):
                    dest = connections_list[i].strip()
                    weight = float(connections_list[i+1].strip())
                    st.session_state.edges[node].append((dest, weight))
                    if not st.session_state.graph.has_edge(node, dest):
                        st.session_state.graph.add_edge(node, dest, weight=weight)
                        if "Undirected" in graph_type:
                            st.session_state.graph.add_edge(dest, node, weight=weight)
            else:
                dests = [dest.strip() for dest in connections.split(',')]
                st.session_state.edges[node].extend(dests)
                for dest in dests:
                    if not st.session_state.graph.has_edge(node, dest):
                        st.session_state.graph.add_edge(node, dest)
                        if "Undirected" in graph_type:
                            st.session_state.graph.add_edge(dest, node)

    if all(st.session_state.edges[node] for node in st.session_state.nodes):
        st.session_state.input_complete = True

def generate_random_graph(num_nodes, num_edges, weighted=False):
    G = nx.DiGraph() if "Directed" in graph_type else nx.Graph()
    G.add_nodes_from([str(i) for i in range(1, num_nodes + 1)])
    while G.number_of_edges() < num_edges:
        u = str(random.randint(1, num_nodes))
        v = str(random.randint(1, num_nodes))
        if u != v and not G.has_edge(u, v):
            if weighted:
                weight = round(random.uniform(1.0, 10.0), 2)
                G.add_edge(u, v, weight=weight)
                if "Undirected" in graph_type:
                    G.add_edge(v, u, weight=weight)
            else:
                G.add_edge(u, v)
                if "Undirected" in graph_type:
                    G.add_edge(v, u)
    return G

if input_method == "Random Graph Generation":
    st.subheader("Random Graph Generator")
    num_nodes = st.number_input("Enter the number of nodes:", min_value=2, step=1)
    num_edges = st.number_input("Enter the number of edges:", min_value=1, step=1)
    
    if st.button("Generate Random Graph"):
        weighted = "Weights" in graph_type
        st.session_state.graph = generate_random_graph(num_nodes, num_edges, weighted=weighted)
        st.session_state.nodes = list(st.session_state.graph.nodes)
        st.session_state.edges = {node: list(st.session_state.graph.neighbors(node)) for node in st.session_state.nodes}
        st.session_state.input_complete = True
        st.success("Random graph generated successfully!")

if st.session_state.input_complete:
    st.subheader("Connections Summary")
    for node in st.session_state.nodes:
        st.write(f"{node}: {', '.join([f'{conn[0]} ({conn[1]})' if isinstance(conn, tuple) else conn for conn in st.session_state.edges[node]])}")

    st.subheader("Select Start and Goal Nodes")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
    with col2:
        st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

    if st.session_state.start_node and st.session_state.goal_node:
        if st.button("Find Path"):
            st.session_state.path = bfs_search(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node)
            if st.session_state.path:
                st.success(f"Path found: {' -> '.join(st.session_state.path)}")
                st.write("Path as List:", st.session_state.path)
                st.write("Path as Dictionary:", {i: st.session_state.path[i] for i in range(len(st.session_state.path))})
            else:
                st.error("No path found.")

if st.session_state.path or st.session_state.goal_node:
    st.subheader("Full Graph:")
    visualize_graph(st.session_state.graph)

    if st.session_state.path:
        st.subheader("Graph with British Museum Search Path:")
        visualize_graph(st.session_state.graph, st.session_state.path)

if st.button("Reset"):
    st.session_state.graph.clear()
    st.session_state.nodes = []
    st.session_state.edges = {}
    st.session_state.start_node = ""
    st.session_state.goal_node = ""
    st.session_state.path = None
    st.session_state.input_complete = False
