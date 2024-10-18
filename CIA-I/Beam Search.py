import streamlit as st
import networkx as nx
import random
import matplotlib.pyplot as plt

def beam_search(graph, heuristics, start, goal, beam_width):
    queue = [[start]]
    paths = []

    while queue:
        curr_path = queue.pop(0)
        current_node = curr_path[-1]

        if current_node == goal:
            paths.append(curr_path)
            continue

        neighbors = graph.neighbors(current_node)
        new_paths = [curr_path + [neighbor] for neighbor in neighbors if neighbor not in curr_path]

        new_paths.sort(key=lambda path: heuristics[path[-1]])
        queue.extend(new_paths[:beam_width])  

    return paths

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

def beam_search_page():
    st.title("Beam Search")

    if 'graph' not in st.session_state:
        st.session_state.graph = nx.Graph()
    if 'nodes' not in st.session_state:
        st.session_state.nodes = []
    if 'edges' not in st.session_state:
        st.session_state.edges = {}
    if 'heuristics' not in st.session_state:
        st.session_state.heuristics = {}
    if 'start_node' not in st.session_state:
        st.session_state.start_node = ""
    if 'goal_node' not in st.session_state:
        st.session_state.goal_node = ""
    if 'paths' not in st.session_state:
        st.session_state.paths = []
    if 'input_complete' not in st.session_state:
        st.session_state.input_complete = False
    if 'beam_width' not in st.session_state:
        st.session_state.beam_width = 3

    input_method = st.radio("Choose input method:", ["Manual Input", "Random Graph Generation"])

    if input_method == "Manual Input":
        nodes_input = st.text_input("Enter all nodes (comma-separated):", key="nodes_input")
        if st.button("Submit Nodes") and nodes_input:
            st.session_state.nodes = [node.strip() for node in nodes_input.split(',')]
            st.session_state.edges = {node: [] for node in st.session_state.nodes}
            st.session_state.heuristics = {node: 0 for node in st.session_state.nodes}

        if st.session_state.nodes:
            for node in st.session_state.nodes:
                connections = st.text_input(f"Enter connections for {node} (comma-separated):", key=f"conn_{node}")
                
                if st.button(f"Submit Connections for {node}", key=f"btn_{node}") and connections:
                    dests = [dest.strip() for dest in connections.split(',')]
                    st.session_state.edges[node] = dests  # Update connections
                    for dest in dests:
                        if not st.session_state.graph.has_edge(node, dest):
                            st.session_state.graph.add_edge(node, dest)

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
            st.session_state.graph.add_nodes_from(st.session_state.nodes)

            for _ in range(num_edges):
                u, v = random.sample(st.session_state.nodes, 2)
                st.session_state.graph.add_edge(u, v)
                st.session_state.edges[u].append(v)

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

        st.session_state.beam_width = st.number_input("Set Beam Width:", min_value=1, max_value=10, value=3, step=1)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Paths (Beam Search)"):
                st.session_state.paths = beam_search(st.session_state.graph, st.session_state.heuristics, st.session_state.start_node, st.session_state.goal_node, st.session_state.beam_width)
                if st.session_state.paths:
                    st.success(f"Paths found: {', '.join([' -> '.join(path) for path in st.session_state.paths])}")
                    for path in st.session_state.paths:
                        visualize_graph(st.session_state.graph, path)
                else:
                    st.error("No path found.")

    if st.session_state.graph:
        st.subheader("Full Graph:")
        visualize_graph(st.session_state.graph)

    if st.button("Reset"):
        st.session_state.clear()

beam_search_page()
