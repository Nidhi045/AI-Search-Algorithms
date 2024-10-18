import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import random
import heapq

def visualize_graph(G, path=[]):
    plt.figure(figsize=(14, 9))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16, 
            font_color='black', edge_color='gray')

    if path:
        path_edges = list(zip(path, path[1:]))   
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)   
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='green', node_size=2000)  
        
    plt.title("Graph Visualization" if not path else "Graph with Path")
    plt.axis('off')
    st.pyplot(plt)

def british_museum_search():
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

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                st.session_state.path = bfs_search(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node)
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

def dfs_search():
    st.title("Depth-First Search (DFS)")
    
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

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                st.session_state.path = dfs_algorithm(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node)
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

def dfs_algorithm(graph, start, goal):
    stack = [[start]]
    visited = set()

    while stack:
        path = stack.pop()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            neighbors =graph.neighbors(node)
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)

    return None

def bfs_page():
    st.title("Breadth-First Search (BFS)")
    
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

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                st.session_state.path = bfs_search(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node)
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

def hill_climbing_page():
    st.title("Hill Climbing Search")

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
                        weight = st.number_input(f"Enter weight for edge {node} -> {dest}:", value=1, min_value=1, key=f"weight_{node}_{dest}")
                        if not st.session_state.graph.has_edge(node, dest):
                            st.session_state.graph.add_edge(node, dest, weight=weight)

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
                weight = random.randint(1, 10)   
                st.session_state.graph.add_edge(u, v, weight=weight)
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

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                st.session_state.path = hill_climbing_search(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node)
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

def hill_climbing_search(graph, start, goal):
    current_node = start
    path = [current_node]

    while current_node != goal:
        neighbors = sorted(graph.neighbors(current_node))
        
        next_node = None
        min_weight = float('inf')

        for neighbor in neighbors:
            weight = graph[current_node][neighbor].get('weight', 1)   
            if weight < min_weight or (weight == min_weight and neighbor < next_node):
                min_weight = weight
                next_node = neighbor

        if not next_node or next_node in path:
            return None

        current_node = next_node
        path.append(current_node)

    return path

def beam_search_page():
    st.title("Beam Search")

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
                        weight = st.number_input(f"Enter weight for edge {node} -> {dest}:", value=1, min_value=1, key=f"weight_{node}_{dest}")
                        if not st.session_state.graph.has_edge(node, dest):
                            st.session_state.graph.add_edge(node, dest, weight=weight)

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
                weight = random.randint(1, 10)   
                st.session_state.graph.add_edge(u, v, weight=weight)
                st.session_state.edges[u].append(v)
            
            st.session_state.input_complete = True
            st.success("Random graph generated successfully!")

    if st.session_state.input_complete:
        st.subheader("Connections Summary")
        for node in st.session_state.nodes:
            st.write(f"{node}: {', '.join(st.session_state.edges[node])}")

        st.subheader("Select Start and Goal Nodes")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
        with col2:
            st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)
        with col3:
            beam_width = st.number_input("Enter the beam width:", min_value=1, step=1, value=2)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                st.session_state.path = beam_search(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node, beam_width)
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

def beam_search(graph, start, goal, beam_width=2):
    queue = [(start, [start])]  
    
    while queue:
        queue.sort(key=lambda x: x[0])
        queue = queue[:beam_width]
        
        next_queue = []

        for node, path in queue:
            if node == goal:
                return path  

            neighbors = sorted(graph.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in path:  
                    next_queue.append((neighbor, path + [neighbor]))

        queue = next_queue

    return None 

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
                st.session_state.path = oracle_search(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node, heuristic)
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

def branch_and_bound_page():
    st.title("Branch and Bound Search")

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
    if 'edge_costs' not in st.session_state:
        st.session_state.edge_costs = {}   

    input_method = st.radio("Choose input method:", ["Manual Input", "Random Graph Generation"])

    if input_method == "Manual Input":
        if not st.session_state.nodes:
            nodes_input = st.text_input("Enter all nodes (comma-separated):", key="nodes_input")
            if st.button("Submit Nodes") and nodes_input:
                st.session_state.nodes = [node.strip() for node in nodes_input.split(',')]
                st.session_state.edges = {node: [] for node in st.session_state.nodes}

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
                            if dest not in st.session_state.edges[node]:
                                st.session_state.edges[node].append(dest)

                                if not st.session_state.graph.has_edge(node, dest):
                                    st.session_state.graph.add_edge(node, dest)
                                    
                                    st.session_state.edge_costs[(node, dest)] = cost_list[i]
                                    st.session_state.edge_costs[(dest, node)] = cost_list[i]

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
            st.session_state.edges = {node: [] for node in st.session_state.nodes}
            st.session_state.graph.add_nodes_from(st.session_state.nodes)

            st.session_state.edge_costs.clear()   
            
            for _ in range(num_edges):
                u, v = random.sample(st.session_state.nodes, 2)
                cost = random.randint(1, 20)   
                st.session_state.graph.add_edge(u, v)
                st.session_state.edges[u].append(v)
                st.session_state.edges[v].append(u)   
                st.session_state.edge_costs[(u, v)] = cost
                st.session_state.edge_costs[(v, u)] = cost   

                st.write(f"Random edge added: {u} <-> {v}, Cost: {cost}")
            
            st.session_state.input_complete = True
            st.success("Random graph generated successfully!")

    if st.session_state.input_complete:
        st.subheader("Connections and Costs Summary")
        for node in st.session_state.nodes:
            connections = ', '.join(
                f"{dest} (cost: {st.session_state.edge_costs.get((node, dest), 'N/A')})"   
                for dest in set(st.session_state.edges[node])  
            )
            st.write(f"{node}: {connections}")

        st.subheader("Select Start and Goal Nodes")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
        with col2:
            st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                path, cost = branch_and_bound_search(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node, st.session_state.edge_costs)
                if path:
                    st.success(f"Path found: {' -> '.join(path)} with total cost: {cost}")
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

def branch_and_bound_search(graph, start, goal, edge_cost):
    pq = [(0, start, [start])]
    visited = set()

    while pq:
        cost, current_node, path = heapq.heappop(pq)

        if current_node == goal:
            return path, cost  

        if current_node not in visited:
            visited.add(current_node)

            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    new_cost = cost + edge_cost[(current_node, neighbor)]
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

    return None, float('inf') 
def branch_and_bound_with_extension_list_search():
    st.title("Branch and Bound with Extension List")
    
    if 'graph' not in st.session_state:
        st.session_state.graph = nx.Graph()
    if 'nodes' not in st.session_state:
        st.session_state.nodes = []
    if 'edges' not in st.session_state:
        st.session_state.edges = {}
    if 'edge_costs' not in st.session_state:
        st.session_state.edge_costs = {}
    if 'start_node' not in st.session_state:
        st.session_state.start_node = ""
    if 'goal_node' not in st.session_state:
        st.session_state.goal_node = ""
    if 'extension_list' not in st.session_state:
        st.session_state.extension_list = []
    if 'path' not in st.session_state:
        st.session_state.path = None
    if 'input_complete' not in st.session_state:
        st.session_state.input_complete = False

    input_method = st.radio("Choose input method:", ["Manual Input", "Random Graph Generation"])
    if st.session_state.input_complete:
        st.subheader("Connections Summary")
        for node in st.session_state.nodes:
            connections = ', '.join(f"{dest} (cost: {st.session_state.edge_costs.get((node, dest), 'N/A')})" for dest in st.session_state.edges[node])
            st.write(f"{node}: {connections}")

        st.subheader("Select Start and Goal Nodes")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
        with col2:
            st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                st.session_state.path, st.session_state.extension_list = branch_and_bound_with_extension_list_algorithm(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node)
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

            st.subheader("Extension List at Each Step:")
            for step in st.session_state.extension_list:
                st.write(step)

    if st.button("Reset"):
        st.session_state.clear()

def branch_and_bound_with_extension_list_algorithm(graph, start, goal):
    import heapq

    queue = []
    heapq.heappush(queue, (0, [start]))  
    
    min_costs = {start: 0}
    extension_list = []

    while queue:
        current_cost, current_path = heapq.heappop(queue)
        current_node = current_path[-1]

        extension_list.append((current_node, current_path, current_cost))

        if current_node == goal:
            return current_path, extension_list

        for neighbor in graph.neighbors(current_node):
            edge_cost = graph[current_node][neighbor].get('cost', 1)   
            new_cost = current_cost + edge_cost
            new_path = current_path + [neighbor]

            if neighbor not in min_costs or new_cost < min_costs[neighbor]:
                min_costs[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, new_path))

    return None, extension_list   

def branch_and_bound_with_heuristic_search():
    st.title("Branch and Bound with Heuristic Search")
    
    if 'graph' not in st.session_state:
        st.session_state.graph = nx.Graph()  
    if 'nodes' not in st.session_state:
        st.session_state.nodes = []
    if 'edges' not in st.session_state:
        st.session_state.edges = {}
    if 'edge_costs' not in st.session_state:
        st.session_state.edge_costs = {}
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
                st.session_state.edge_costs = {}

        if st.session_state.nodes:
            st.subheader("Enter connections and costs for each node")
            for node in st.session_state.nodes:
                connections = st.text_input(f"Enter connections for {node} (format: destination1:cost1, destination2:cost2):", key=f"conn_{node}")
                if st.button(f"Submit Connections for {node}", key=f"btn_{node}") and connections:
                    dests_costs = [conn.split(':') for conn in connections.split(',')]
                    for dest, cost in dests_costs:
                        dest = dest.strip()
                        cost = int(cost.strip())
                        st.session_state.edges[node].append(dest)
                        st.session_state.edge_costs[(node, dest)] = cost
                        if not st.session_state.graph.has_edge(node, dest):
                            st.session_state.graph.add_edge(node, dest, cost=cost)

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
            st.session_state.graph.add_nodes_from(st.session_state.nodes)
            
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
            connections = ', '.join(f"{dest} (cost: {st.session_state.edge_costs[(node, dest)]})" for dest in st.session_state.edges[node])
            st.write(f"{node}: {connections}")

        st.subheader("Select Start and Goal Nodes")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
        with col2:
            st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                st.session_state.path, st.session_state.extension_list = branch_and_bound_with_heuristic_algorithm(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node)
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
        st.subheader("Extension List:")
        for node, path, cost in st.session_state.extension_list:
            st.write(f"Node: {node}, Path: {' -> '.join(path)}, Cost: {cost}")

    if st.button("Reset"):
        st.session_state.clear()

def branch_and_bound_with_heuristic_algorithm(graph, start_node, goal_node):
    queue = []
    heapq.heappush(queue, (0, start_node, [start_node]))   
    visited = set()
    best_path = None
    best_cost = float('inf')
    
    extension_list = []   

    while queue:
        current_cost, current_node, path = heapq.heappop(queue)
        
        if current_node == goal_node:
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = path
            continue
        
        visited.add(current_node)

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                edge_cost = graph[current_node][neighbor]['cost']
                new_cost = current_cost + edge_cost
                heuristic = estimate_heuristic(neighbor, goal_node)
                estimated_total_cost = new_cost + heuristic
                extension_list.append((neighbor, path + [neighbor], new_cost))
                heapq.heappush(queue, (estimated_total_cost, neighbor, path + [neighbor]))

    return best_path, extension_list

def estimate_heuristic(node, goal_node):
    return abs(int(node) - int(goal_node)) 

def a_star_algorithm(graph, start, goal, heuristic):
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
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic[neighbor]
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current_node

    return None, extension_list

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
                st.session_state.heuristic = {node: 0 for node in st.session_state.nodes}  # Initialize heuristic
                
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
                    
                    # Set heuristic for the destination node
                    heuristic_value = st.number_input(f"Enter heuristic for {dest.strip()} (or 0):", value=0.0, key=f"heur_{dest.strip()}")
                    st.session_state.heuristic[dest.strip()] = heuristic_value

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
            if st.button("Find Path"):
                st.session_state.path, st.session_state.extension_list = a_star_algorithm(
                    st.session_state.graph,
                    st.session_state.start_node,
                    st.session_state.goal_node,
                    st.session_state.heuristic
                )
                if st.session_state.path:
                    st.success(f"Path found: {' -> '.join(st.session_state.path)}")
                else:
                    st.error("No path found.")

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

def ao_star_algorithm(graph, start, goal):
    open_set = [(start, 0)]
    best_costs = {start: 0}
    paths = {start: [start]}
    extension_list = []

    while open_set:
        current_node, current_cost = open_set.pop(0)
        extension_list.append((current_node, current_cost))

        if current_node == goal:
            return paths[current_node], extension_list

        for neighbor in graph.neighbors(current_node):
            edge_cost = graph[current_node][neighbor]['cost']
            total_cost = current_cost + edge_cost
            
            if neighbor not in best_costs or total_cost < best_costs[neighbor]:
                best_costs[neighbor] = total_cost
                paths[neighbor] = paths[current_node] + [neighbor]
                open_set.append((neighbor, total_cost))
                open_set = sorted(open_set, key=lambda x: best_costs[x[0]])  # Sort based on costs

    return None, extension_list

def ao_star_search_page():
    st.title("AO* Search Algorithm")
    
    if 'graph' not in st.session_state:
        st.session_state.graph = nx.Graph()  
    if 'nodes' not in st.session_state:
        st.session_state.nodes = []
    if 'edges' not in st.session_state:
        st.session_state.edges = {}
    if 'edge_costs' not in st.session_state:
        st.session_state.edge_costs = {}
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
                st.session_state.edge_costs = {}
                
        if st.session_state.nodes:
            st.subheader("Enter connections and costs for each node")
            for node in st.session_state.nodes:
                connections = st.text_input(f"Enter connections and costs for {node} (format: destination,cost):", key=f"conn_{node}")
                if st.button(f"Submit Connections for {node}", key=f"btn_{node}") and connections:
                    dest_costs = [x.strip().split(',') for x in connections.split(';')]
                    for dest_cost in dest_costs:
                        if len(dest_cost) == 2:
                            dest, cost = dest_cost
                            cost = float(cost.strip())
                            st.session_state.edges[node].append(dest.strip())
                            st.session_state.graph.add_edge(node.strip(), dest.strip(), cost=cost)
                            st.session_state.edge_costs[(node.strip(), dest.strip())] = cost
                    st.success(f"Connections for {node} added successfully.")

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

        st.subheader("Select Start and Goal Nodes")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
        with col2:
            st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                st.session_state.path, st.session_state.extension_list = ao_star_algorithm(
                    st.session_state.graph,
                    st.session_state.start_node,
                    st.session_state.goal_node
                )
                if st.session_state.path:
                    st.success(f"Path found: {' -> '.join(st.session_state.path)}")
                else:
                    st.error("No path found.")

                st.subheader("Extension List at Each Step:")
                for ext in st.session_state.extension_list:
                    st.write(f"Node: {ext[0]}, Cost: {ext[1]}")

    if st.session_state.path or st.session_state.goal_node:
        st.subheader("Full Graph:")
        visualize_graph(st.session_state.graph)

        if st.session_state.path:
            st.subheader("Graph with Path:")
            visualize_graph(st.session_state.graph, st.session_state.path)

    if st.button("Reset"):
        st.session_state.clear()

def best_first_search_algorithm(graph, start, goal, heuristics):
    open_set = [(heuristics[start], start)]
    best_costs = {start: 0}
    paths = {start: [start]}
    extension_list = []

    while open_set:
        _, current_node = heapq.heappop(open_set)
        extension_list.append((current_node, best_costs[current_node]))

        if current_node == goal:
            return paths[current_node], extension_list

        for neighbor in graph.neighbors(current_node):
            edge_cost = graph[current_node][neighbor]['cost']
            total_cost = best_costs[current_node] + edge_cost
            
            if neighbor not in best_costs or total_cost < best_costs[neighbor]:
                best_costs[neighbor] = total_cost
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(open_set, (heuristics[neighbor], neighbor))

    return None, extension_list

def best_first_search_page():
    st.title("Best-First Search Algorithm")
    
    if 'graph' not in st.session_state:
        st.session_state.graph = nx.Graph()  
    if 'nodes' not in st.session_state:
        st.session_state.nodes = []
    if 'edges' not in st.session_state:
        st.session_state.edges = {}
    if 'edge_costs' not in st.session_state:
        st.session_state.edge_costs = {}
    if 'start_node' not in st.session_state:
        st.session_state.start_node = ""
    if 'goal_node' not in st.session_state:
        st.session_state.goal_node = ""
    if 'heuristics' not in st.session_state:
        st.session_state.heuristics = {}
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
                st.session_state.edge_costs = {}
                
        if st.session_state.nodes:
            st.subheader("Enter connections and costs for each node")
            for node in st.session_state.nodes:
                connections = st.text_input(f"Enter connections and costs for {node} (format: destination,cost):", key=f"conn_{node}")
                if st.button(f"Submit Connections for {node}", key=f"btn_{node}") and connections:
                    dest_costs = [x.strip().split(',') for x in connections.split(';')]
                    for dest_cost in dest_costs:
                        if len(dest_cost) == 2:
                            dest, cost = dest_cost
                            cost = float(cost.strip())
                            st.session_state.edges[node].append(dest.strip())
                            st.session_state.graph.add_edge(node.strip(), dest.strip(), cost=cost)
                            st.session_state.edge_costs[(node.strip(), dest.strip())] = cost
                    st.success(f"Connections for {node} added successfully.")

            if all(st.session_state.edges[node] for node in st.session_state.nodes):
                st.session_state.input_complete = True

        if st.session_state.input_complete:
            st.subheader("Enter Heuristics for Each Node")
            for node in st.session_state.nodes:
                heuristic = st.number_input(f"Enter heuristic for {node}:", key=f"heuristic_{node}", min_value=0.0)
                st.session_state.heuristics[node] = heuristic
            
            st.subheader("Connections and Costs Summary")
            for node in st.session_state.nodes:
                connections = ', '.join(f"{dest} (cost: {st.session_state.edge_costs.get((node, dest), 'N/A')})" for dest in st.session_state.edges[node])
                st.write(f"{node}: {connections}")

            st.subheader("Select Start and Goal Nodes")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.start_node = st.selectbox("Select the starting node:", st.session_state.nodes)
            with col2:
                st.session_state.goal_node = st.selectbox("Select the goal node:", st.session_state.nodes)

            if st.session_state.start_node and st.session_state.goal_node:
                if st.button("Find Path"):
                    st.session_state.path, st.session_state.extension_list = best_first_search_algorithm(
                        st.session_state.graph,
                        st.session_state.start_node,
                        st.session_state.goal_node,
                        st.session_state.heuristics
                    )
                    if st.session_state.path:
                        st.success(f"Path found: {' -> '.join(st.session_state.path)}")
                    else:
                        st.error("No path found.")

                    st.subheader("Extension List at Each Step:")
                    for ext in st.session_state.extension_list:
                        st.write(f"Node: {ext[0]}, Cost: {ext[1]}")

    elif input_method == "Random Graph Generation":
        st.subheader("Random Graph Generator")
        num_nodes = st.number_input("Enter the number of nodes:", min_value=2, step=1)
        num_edges = st.number_input("Enter the number of edges:", min_value=1, step=1)

        if st.button("Generate Random Graph"):
            st.session_state.graph.clear()
            st.session_state.nodes = [str(i) for i in range(1, num_nodes + 1)]
            st.session_state.edges = {node: [] for node in st.session_state.nodes}
            st.session_state.edge_costs = {}
            st.session_state.heuristics = {node: random.randint(1, 10) for node in st.session_state.nodes}
            
            for _ in range(num_edges):
                u, v = random.sample(st.session_state.nodes, 2)
                cost = random.randint(1, 10)
                st.session_state.graph.add_edge(u, v, cost=cost)
                st.session_state.edges[u].append(v)
                st.session_state.edge_costs[(u, v)] = cost
            
            st.session_state.input_complete = True
            st.success("Random graph generated successfully!")

    if st.session_state.path or st.session_state.goal_node:
        st.subheader("Full Graph:")
        visualize_graph(st.session_state.graph)

        if st.session_state.path:
            st.subheader("Graph with Path:")
            visualize_graph(st.session_state.graph, st.session_state.path)

    if st.button("Reset"):
        st.session_state.clear()

def main():
    pages = {
        "British Museum Search": british_museum_search, # uninformed
        "Depth-First Search (DFS)": dfs_search, # uninformed
        "Breadth-First Search (BFS)": bfs_page, # uninformed
        "Hill Climbing Search": hill_climbing_page, # heuristics 
        "Beam Search": beam_search_page, # heuristics 
        "Oracle Search": oracle_search_page,
        "Branch and Bound Search": branch_and_bound_page,
        "Branch and Bound with Extension List": branch_and_bound_with_extension_list_search,
        "Branch and Bound with Heuristic": branch_and_bound_with_heuristic_search,
        "A* Search": a_star_search_page, # optimal 
        "AO* Search": ao_star_search_page,
        "Best-First Search": best_first_search_page,
    }

    selection = st.sidebar.selectbox("Select Search Algorithm", list(pages.keys()))
    pages[selection]()       

if __name__ == "__main__":
    main()