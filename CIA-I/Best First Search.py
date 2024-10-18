import streamlit as st
import networkx as nx
import random
import matplotlib.pyplot as plt
from queue import PriorityQueue
import time

def best_first_search_page():
    st.title("Best First Search")

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
    if 'input_complete' not in st.session_state:
        st.session_state.input_complete = False
    if 'heuristics' not in st.session_state:
        st.session_state.heuristics = {}

    input_method = st.radio("Choose input method:", ["Manual Input", "Random Graph Generation"])

    if input_method == "Manual Input":
        if not st.session_state.nodes:
            nodes_input = st.text_input("Enter all nodes (comma-separated):", key="nodes_input")
            if st.button("Submit Nodes") and nodes_input:
                st.session_state.nodes = [node.strip() for node in nodes_input.split(',')]
                st.session_state.edges = {node: {} for node in st.session_state.nodes}   
                st.session_state.heuristics = {node: 0 for node in st.session_state.nodes}

        if st.session_state.nodes:
            st.subheader("Enter connections and costs for each node")
            for node in st.session_state.nodes:
                connections = st.text_input(f"Enter connections for {node}:", key=f"conn_{node}")
                costs = st.text_input(f"Enter corresponding costs for {node}'s connections (comma-separated):", key=f"cost_{node}")
                heuristic = st.number_input(f"Enter heuristic for {node}:", key=f"heuristic_{node}")

                if st.button(f"Submit Connections for {node}", key=f"btn_{node}") and connections and costs:
                    dests = [dest.strip() for dest in connections.split(',')]
                    cost_list = [float(cost.strip()) for cost in costs.split(',')]

                    if len(dests) == len(cost_list):
                        for i, dest in enumerate(dests):
                            st.session_state.edges[node][dest] = cost_list[i]  
                            st.session_state.graph.add_edge(node, dest)
                            st.write(f"Edge added: {node} <-> {dest}, Cost: {cost_list[i]}")
                        st.session_state.heuristics[node] = heuristic
                    else:
                        st.error("Mismatch between number of connections and costs. Please correct the inputs.")

            if all(st.session_state.edges[node] for node in st.session_state.nodes):
                st.session_state.input_complete = True

    elif input_method == "Random Graph Generation":
        st.subheader("Random Graph Generator")
        num_nodes = st.number_input("Enter the number of nodes:", min_value=2, step=1)
        num_edges = st.number_input("Enter the number of edges:", min_value=1, step=1)

        if st.button("Generate Random Graph"):
            st.session_state.graph.clear()
            st.session_state.nodes = [str(i) for i in range(1, num_nodes + 1)]
            st.session_state.edges = {node: {} for node in st.session_state.nodes}
            st.session_state.heuristics = {node: random.randint(1, 10) for node in st.session_state.nodes}

            for _ in range(num_edges):
                u, v = random.sample(st.session_state.nodes, 2)
                cost = random.randint(1, 20)
                st.session_state.graph.add_edge(u, v)
                st.session_state.edges[u][v] = cost   
                st.session_state.edges[v][u] = cost   
                st.write(f"Random edge added: {u} <-> {v}, Cost: {cost}")

            st.session_state.input_complete = True
            st.success("Random graph generated successfully!")

    if st.session_state.input_complete:
        st.subheader("Connections and Costs Summary")
        for node in st.session_state.nodes:
            connections = ', '.join(
                f"{dest} (cost: {st.session_state.edges[node][dest]})"
                for dest in st.session_state.edges[node]
            )
            st.write(f"{node}: {connections}, Heuristic: {st.session_state.heuristics[node]}")

        st.subheader("Select Start and Goal Nodes")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
        with col2:
            st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                paths = best_first_search(st.session_state.edges, st.session_state.start_node, st.session_state.goal_node, st.session_state.heuristics)
                if paths:
                    for path in paths:
                        st.success(f"Exploring path: {' -> '.join(path)}")
                        update_graph_visualization(st.session_state.graph, path)
                        time.sleep(1)   
                else:
                    st.error("No path found.")

    if st.session_state.graph:
        st.subheader("Full Graph:")
        visualize_graph(st.session_state.graph)

def best_first_search(edges, start, goal, heuristics):
    queue = PriorityQueue()
    queue.put((heuristics[start], [start]))   
    explored = set()
    results = []

    while not queue.empty():
        _, path = queue.get()
        current_node = path[-1]

        explored.add(current_node)

        if current_node == goal:
            results.append(path)
            continue

        for neighbor, edge_cost in edges[current_node].items():
            if neighbor not in explored:
                new_path = path + [neighbor]
                new_cost = heuristics[neighbor]   
                queue.put((new_cost, new_path))

    return results

def visualize_graph(graph):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
    st.pyplot(plt)

def update_graph_visualization(graph, path):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)

    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2)

    st.pyplot(plt)
    
best_first_search_page()
