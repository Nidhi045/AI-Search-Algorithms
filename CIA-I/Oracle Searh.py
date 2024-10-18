import streamlit as st
import networkx as nx
import random
import time
import matplotlib.pyplot as plt
from collections import deque

def oracle_search_page():
    st.title("Oracle Search")

    if 'graph' not in st.session_state:
        st.session_state.graph = nx.Graph()  
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

    input_method = st.radio("Choose input method:", ["Manual Input", "Random Graph Generation"])
    
    # Manual Input
    if input_method == "Manual Input":
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
                    dests = [dest.strip() for dest in connections.split(',')]
                    st.session_state.edges[node].extend(dests)
                    for dest in dests:
                        if not st.session_state.graph.has_edge(node, dest):
                            st.session_state.graph.add_edge(node, dest)

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

        st.subheader("Select Start and Goal Nodes")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
        with col2:
            st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

        st.subheader("Enter heuristic values for each node:")
        heuristic = {}
        for node in st.session_state.nodes:
            heuristic[node] = st.number_input(f"Heuristic value for {node}:", value=0, min_value=0)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                st.session_state.path = []
                with st.spinner("Finding path..."):
                    st.session_state.path = oracle_search(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node, heuristic)
                    if st.session_state.path:
                        st.success(f"Path found: {' -> '.join(st.session_state.path)}")
                    else:
                        st.error("No path found.")

                visualize_path(st.session_state.graph, st.session_state.path)

    if st.button("Reset"):
        st.session_state.clear()

def oracle_search(graph, start, goal, heuristic):
    queue = deque([(start, [start])])   
    visited = set()

    while queue:
        queue = deque(sorted(queue, key=lambda x: heuristic[x[0]]))   
        current_node, path = queue.popleft()

        if current_node == goal:
            return path   

        if current_node not in visited:
            visited.add(current_node)
            neighbors = graph.neighbors(current_node)

            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

    return None  

def visualize_path(graph, path):
    if path:
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            st.write(f"Traversing: {edge[0]} -> {edge[1]}")
            time.sleep(1)   
            
            visualize_graph(graph, path[:i + 2])  

    visualize_graph(graph)

def visualize_graph(graph, path=None):
    fig, ax = plt.subplots()
    pos = nx.spring_layout(graph)  
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16, font_color='black', edge_color='gray')

    if path: 
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2)
        
    st.pyplot(fig)   

if __name__ == "__main__":
    oracle_search_page()
