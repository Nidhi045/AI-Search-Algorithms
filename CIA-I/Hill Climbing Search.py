import streamlit as st
import networkx as nx
import random
import matplotlib.pyplot as plt

def hill_climbing_search(graph, heuristics, start, goal):
    current_node = start
    path = [current_node]
    visited = set(path)

    while current_node != goal:
        neighbors = sorted(graph.neighbors(current_node), key=lambda n: heuristics[n])
        next_node = None
        min_heuristic = float('inf')

        for neighbor in neighbors:
            if neighbor not in visited:
                heuristic = heuristics[neighbor]
                if heuristic < min_heuristic:
                    min_heuristic = heuristic
                    next_node = neighbor

        if next_node is None:  
            return None

        visited.add(next_node)
        path.append(next_node)
        current_node = next_node

    return path

def visualize_graph(graph, path=None):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=15, font_weight='bold')

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='orange', width=2)

    plt.title("Graph Visualization")
    plt.axis('off')
    st.pyplot(plt)

def hill_climbing_page():
    st.title("Hill Climbing Search")

    if 'graph' not in st.session_state:
        st.session_state.graph = nx.Graph()
    if 'nodes' not in st.session_state:
        st.session_state.nodes = []
    if 'edges' not in st.session_state:
        st.session_state.edges = {}
    if 'weights' not in st.session_state:
        st.session_state.weights = {}
    if 'heuristics' not in st.session_state:
        st.session_state.heuristics = {}
    if 'start_node' not in st.session_state:
        st.session_state.start_node = ""
    if 'goal_node' not in st.session_state:
        st.session_state.goal_node = ""
    if 'path' not in st.session_state:
        st.session_state.path = None
    if 'input_complete' not in st.session_state:
        st.session_state.input_complete = False

    input_method = st.radio("Choose input method:", ["Manual Input", "Random Graph Generation"])
    
    if input_method == "Manual Input":
        nodes_input = st.text_input("Enter all nodes (comma-separated):", key="nodes_input")
        if st.button("Submit Nodes") and nodes_input:
            st.session_state.nodes = [node.strip() for node in nodes_input.split(',')]
            st.session_state.edges = {node: [] for node in st.session_state.nodes}
            st.session_state.weights = {node: {} for node in st.session_state.nodes}
            st.session_state.heuristics = {node: 0 for node in st.session_state.nodes}

        if st.session_state.nodes:
            for node in st.session_state.nodes:
                connections = st.text_input(f"Enter connections for {node} (comma-separated):", key=f"conn_{node}", value=", ".join(st.session_state.edges[node]))
                
                if st.button(f"Submit Connections for {node}", key=f"btn_{node}") and connections:
                    dests = [dest.strip() for dest in connections.split(',')]
                    st.session_state.edges[node] = dests  # Update connections
                    for dest in dests:
                        if not st.session_state.graph.has_edge(node, dest):
                            st.session_state.graph.add_edge(node, dest)

                # Keep the weights input visible
                for dest in st.session_state.edges[node]:
                    weight_key = f"weight_{node}_{dest}"
                    weight = st.number_input(f"Enter weight for edge {node} -> {dest}:", value=st.session_state.weights.get(weight_key, 1), min_value=1, key=weight_key)
                    st.session_state.weights[weight_key] = weight

            st.subheader("Enter heuristic values for each node")
            for node in st.session_state.nodes:
                heuristic_value = st.number_input(f"Enter heuristic value for {node}:", value=st.session_state.heuristics[node], key=f"heur_{node}")
                st.session_state.heuristics[node] = heuristic_value

            if all(st.session_state.edges[node] for node in st.session_state.nodes):
                st.session_state.input_complete = True

    elif input_method == "Random Graph Generation":
        st.subheader("Random Graph Generator")
        num_nodes = st.number_input("Enter the number of nodes:", min_value=2, step=1)
        num_edges = st.number_input("Enter the number of edges:", min_value=1, step=1)

        if st.button("Generate Random Graph"):
            st.session_state.graph.clear()
            st.session_state.nodes = [str(i) for i in range(1, num_nodes + 1)]
            st.session_state.edges = {node: [] for node in st.session_state.nodes}
            st.session_state.heuristics = {node: random.uniform(0, 10) for node in st.session_state.nodes}
            st.session_state.weights = {node: {} for node in st.session_state.nodes}
            st.session_state.graph.add_nodes_from(st.session_state.nodes)

            for _ in range(num_edges):
                u, v = random.sample(st.session_state.nodes, 2)
                weight = random.randint(1, 10)
                st.session_state.graph.add_edge(u, v, weight=weight)
                st.session_state.edges[u].append(v)
                st.session_state.weights[f"weight_{u}_{v}"] = weight  # Store the generated weights

            st.session_state.input_complete = True
            st.success("Random graph generated successfully!")

    if st.session_state.input_complete:
        st.subheader("Connections Summary")
        for node in st.session_state.nodes:
            st.write(f"{node}: {', '.join(st.session_state.edges[node])}")
            st.write(f"Heuristic: {st.session_state.heuristics[node]}")

        st.subheader("Select Start and Goal Nodes")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
        with col2:
            st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path (Hill Climbing)"):
                st.session_state.path = hill_climbing_search(st.session_state.graph, st.session_state.heuristics, st.session_state.start_node, st.session_state.goal_node)
                if st.session_state.path:
                    st.success(f"Path found: {' -> '.join(st.session_state.path)}")
                else:
                    st.error("No path found.")

    if st.session_state.path or st.session_state.goal_node:
        st.subheader("Full Graph:")
        visualize_graph(st.session_state.graph)

        if st.session_state.path:
            st.subheader("Graph with Path:")
            visualize_graph(st.session_state.graph, st.session_state.path)

    if st.button("Reset"):
        st.session_state.clear()

hill_climbing_page()
