import streamlit as st
import networkx as nx
import heapq
import random
import matplotlib.pyplot as plt

def a_star_algorithm(graph, start, goal, heuristic, oracle_cost):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    extension_list = []

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        extension_list.append({
            'node': current_node,
            'path': ' -> '.join(came_from.get(current_node, [start])),
            'cost': cost_so_far[current_node],
            'heuristic': heuristic[current_node]
        })

        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path.reverse()
            return path, extension_list

        for neighbor in graph.neighbors(current_node):
            new_cost = cost_so_far[current_node] + graph[current_node][neighbor]['cost']
            total_cost = new_cost + heuristic[neighbor]

            if total_cost <= oracle_cost and (neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]):
                cost_so_far[neighbor] = new_cost
                heapq.heappush(open_set, (total_cost, neighbor))
                came_from[neighbor] = current_node

    return None, extension_list

def visualize_graph(graph, path=None):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightgray', node_size=2000, font_size=16)
    edge_labels = nx.get_edge_attributes(graph, 'cost')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=3)
    
    plt.title("Graph Visualization")
    st.pyplot(plt)

def a_star_search_page():
    st.title("A* Search Algorithm")
    
    if 'graph' not in st.session_state:
        st.session_state.graph = nx.Graph()  
    if 'nodes' not in st.session_state:
        st.session_state.nodes = []
    if 'edges' not in st.session_state:
        st.session_state.edges = {}
    if 'edge_costs' not in st.session_state:
        st.session_state.edge_costs = {}
    if 'heuristic' not in st.session_state:
        st.session_state.heuristic = {}
    if 'start_node' not in st.session_state:
        st.session_state.start_node = ""
    if 'goal_node' not in st.session_state:
        st.session_state.goal_node = ""
    if 'path' not in st.session_state:
        st.session_state.path = None
    if 'extension_list' not in st.session_state:
        st.session_state.extension_list = []
    if 'input_complete' not in st.session_state:
        st.session_state.input_complete = False

    input_method = st.radio("Choose input method:", ["Manual Input", "Random Graph Generation"])
    
    if input_method == "Manual Input":
        if not st.session_state.nodes:
            nodes_input = st.text_input("Enter all nodes (comma-separated):", key="nodes_input")
            if st.button("Submit Nodes") and nodes_input:
                st.session_state.nodes = [node.strip() for node in nodes_input.split(',')]
                st.session_state.edges = {node: [] for node in st.session_state.nodes}
                st.session_state.heuristic = {node: None for node in st.session_state.nodes}  # Initialize heuristic as None
                
        if st.session_state.nodes:
            st.subheader("Enter connections and costs for each node")
            for node in st.session_state.nodes:
                connections = st.text_input(f"Enter connections and costs for {node} (format: destination,cost):", key=f"conn_{node}")
                if st.button(f"Submit Connections for {node}", key=f"btn_{node}") and connections:
                    dest_costs = [x.strip().split(',') for x in connections.split(';')]
                    for dest, cost in dest_costs:
                        cost = float(cost.strip())
                        st.session_state.edges[node].append(dest.strip())
                        st.session_state.graph.add_edge(node.strip(), dest.strip(), cost=cost)
                        st.session_state.edge_costs[(node.strip(), dest.strip())] = cost
                    
            st.subheader("Enter Heuristic Values for Nodes")
            for node in st.session_state.nodes:
                heuristic_value = st.number_input(f"Enter heuristic for {node} (or 0):", value=st.session_state.heuristic[node] if st.session_state.heuristic[node] is not None else 0.0, key=f"heur_{node}")
                st.session_state.heuristic[node] = heuristic_value

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
            st.session_state.edge_costs = {}
            st.session_state.heuristic = {node: random.randint(1, 10) for node in st.session_state.nodes}  # Random heuristic
            
            for _ in range(num_edges):
                u, v = random.sample(st.session_state.nodes, 2)
                cost = random.randint(1, 10)
                st.session_state.graph.add_edge(u, v, cost=cost)
                st.session_state.edges[u].append(v)
                st.session_state.edge_costs[(u, v)] = cost
            
            st.session_state.input_complete = True
            st.success("Random graph generated successfully!")

    if st.session_state.input_complete:
        st.subheader("Connections and Costs Summary")
        for node in st.session_state.nodes:
            connections = ', '.join(f"{dest} (cost: {st.session_state.edge_costs.get((node, dest), 'N/A')})" for dest in st.session_state.edges[node])
            st.write(f"{node}: {connections}")

        st.subheader("Heuristics Summary")
        for node in st.session_state.nodes:
            st.write(f"Heuristic for {node}: {st.session_state.heuristic[node]}")

        st.subheader("Select Start and Goal Nodes")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
        with col2:
            st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

        if st.session_state.start_node and st.session_state.goal_node:
            oracle_cost = st.number_input("Enter oracle cost:", value=30.0)
            if st.button("Find Path"):
                st.session_state.path, st.session_state.extension_list = a_star_algorithm(
                    st.session_state.graph,
                    st.session_state.start_node,
                    st.session_state.goal_node,
                    st.session_state.heuristic,
                    oracle_cost
                )
                if st.session_state.path:
                    st.success(f"Path found: {' -> '.join(st.session_state.path)}")
                else:
                    st.error("No path found within oracle cost.")

                st.subheader("Extension List at Each Step:")
                for ext in st.session_state.extension_list:
                    st.write(f"Node: {ext['node']}, Path: {ext['path']}, Cost: {ext['cost']}, Heuristic: {ext['heuristic']}")

    if st.session_state.path or st.session_state.goal_node:
        st.subheader("Full Graph:")
        visualize_graph(st.session_state.graph)

        if st.session_state.path:
            st.subheader("Graph with Path:")
            visualize_graph(st.session_state.graph, st.session_state.path)

    if st.button("Reset"):
        st.session_state.clear() 

if __name__ == "__main__":
    a_star_search_page()
