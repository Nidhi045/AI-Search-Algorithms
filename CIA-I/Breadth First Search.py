import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import time

def bfs_search(graph, start_node, goal_node):
    queue = [[start_node]]  
    visited = set()          
    while queue:
        path = queue.pop(0)   
        node = path[-1]      

        if node == goal_node:
            return path   

        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                new_path = list(path)   
                new_path.append(neighbor)
                queue.append(new_path)   

    return None  

def visualize_graph(graph, path=[], node_colors="lightblue"):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(6, 4))
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, font_weight="bold", edge_color="gray")

    if path:
        edges_in_path = [(path[i], path[i+1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=edges_in_path, width=2.5, edge_color="red")
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color="orange", node_size=600)

    st.pyplot(plt)

def bfs_page():
    st.title("Breadth-First Search (BFS)")
    with st.expander("More About BFS", expanded=False):
        st.info("""
        **BFS Algorithm Overview:**
        - **Initialization**: BFS starts at the root (or an arbitrary node) and explores all its neighbors.
        - **Queue**: It uses a queue to keep track of the nodes that need to be explored next.
        - **Visited Set**: A set is used to keep track of visited nodes to avoid processing the same node multiple times.

        **Steps of the BFS Algorithm**:
        1. Start by enqueuing the root node and marking it as visited.
        2. Dequeue a node from the front of the queue.
        3. Process that node (check if it's the goal node).
        4. Enqueue all unvisited neighbors of the dequeued node and mark them as visited.
        5. Repeat steps 2-4 until the queue is empty or the goal node is found.

        **Applications of BFS**:
        - Finding the shortest path in an unweighted graph.
        - Peer-to-peer networks.
        - Crawlers in search engines.
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
                path = bfs_search(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node)

                if path:
                    for i in range(len(path)):
                        st.write(f"Exploring path: {' -> '.join(path[:i+1])}")
                        visualize_graph(st.session_state.graph, path[:i+1])
                        time.sleep(1)   
                    
                    st.success(f"Final Path found: {' -> '.join(path)}")
                else:
                    st.error("No path found.")


st.sidebar.title("BFS Algorithm")
bfs_page()
