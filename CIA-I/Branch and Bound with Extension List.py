import streamlit as st
import networkx as nx
import random
import matplotlib.pyplot as plt

def branch_and_bound_page():
    st.title("Branch and Bound Search with Extended List")

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

    input_method = st.radio("Choose input method:", ["Manual Input", "Random Graph Generation"])

    if input_method == "Manual Input":
        if not st.session_state.nodes:
            nodes_input = st.text_input("Enter all nodes (comma-separated):", key="nodes_input")
            if st.button("Submit Nodes") and nodes_input:
                st.session_state.nodes = [node.strip() for node in nodes_input.split(',')]
                st.session_state.edges = {node: {} for node in st.session_state.nodes}   

        if st.session_state.nodes:
            st.subheader("Enter connections and costs for each node")
            for node in st.session_state.nodes:
                connections = st.text_input(f"Enter connections for {node}:", key=f"conn_{node}")
                costs = st.text_input(f"Enter corresponding costs for {node}'s connections (comma-separated):", key=f"cost_{node}")

                if st.button(f"Submit Connections for {node}", key=f"btn_{node}") and connections and costs:
                    dests = [dest.strip() for dest in connections.split(',')]
                    cost_list = [float(cost.strip()) for cost in costs.split(',')]

                    if len(dests) == len(cost_list):
                        for i, dest in enumerate(dests):
                            st.session_state.edges[node][dest] = cost_list[i]  
                            st.session_state.graph.add_edge(node, dest, weight=cost_list[i])  
                            st.write(f"Edge added: {node} <-> {dest}, Cost: {cost_list[i]}")
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

            for _ in range(num_edges):
                u, v = random.sample(st.session_state.nodes, 2)
                cost = random.randint(1, 20)
                st.session_state.graph.add_edge(u, v, weight=cost)  
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
            st.write(f"{node}: {connections}")

        st.subheader("Select Start and Goal Nodes")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
        with col2:
            st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

        oracle_value = st.number_input("Enter the Oracle value:", min_value=0)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                path_steps = []  
                path, cost = branch_and_bound_extended_list(st.session_state.edges, st.session_state.start_node, st.session_state.goal_node, oracle_value)
                if path:
                    st.success(f"Path found: {' -> '.join(path)} with total cost: {cost}")
                else:
                    st.error("No path found.")

                if path_steps:
                    st.subheader("Path Exploration Steps")
                    for step in path_steps:
                        visualize_graph(st.session_state.graph, step[1])  

    if st.session_state.graph:
        st.subheader("Full Graph:")
        visualize_graph(st.session_state.graph)

def branch_and_bound_extended_list(edges, start, goal, oracle):
    queue = [[0, start, [start]]]  # (cost, current_node, path)
    visited = set()

    st.write("Starting Branch and Bound with Extended List search...")
    st.write(f"Start node: {start}")
    st.write(f"Goal node: {goal}")
    st.write(f"Oracle value: {oracle}")

    while queue:
        current_cost, current_node, current_path = queue.pop(0)

        st.write(f"\nExploring node: {current_node}")
        st.write(f"Current path: {current_path}")
        st.write(f"Visited nodes: {visited}")

        if current_node == goal:
            st.write("Goal found!")
            return current_path, current_cost

        if current_node not in visited:  
            visited.add(current_node)

            successors = []
            for neighbor, edge_cost in edges[current_node].items():
                if neighbor not in visited:
                    new_cost = current_cost + edge_cost
                    if new_cost <= oracle:
                        new_path = current_path + [neighbor]  # Update path correctly
                        successors.append((new_cost, neighbor, new_path))

            successors.sort(key=lambda x: x[0])  # Sort by cost
            queue.extend(successors)

            st.write(f"Generated successors: {successors}")
            st.write(f"Updated queue: {queue}")

            # Visualize the current path taken
            visualize_graph(st.session_state.graph, current_path)

    st.write("No path found within oracle cost!")
    return None, float('inf')

def visualize_graph(graph, path=None):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
    
    if path:
        edge_labels = {(u, v): f"{graph[u][v]['weight']}" for u, v in graph.edges() if 'weight' in graph[u][v]}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        valid_path_edges = [edge for edge in path_edges if edge in graph.edges()]
        nx.draw_networkx_edges(graph, pos, edgelist=valid_path_edges, edge_color='red', width=2)
    
    st.pyplot(plt)

branch_and_bound_page()
