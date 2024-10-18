import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import random

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

def bms_search():
    st.title("British Museum Search (BMS)")

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
    if 'max_goals' not in st.session_state:
        st.session_state.max_goals = 1
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

        st.session_state.max_goals = st.slider("Select max number of goals to find:", min_value=1, max_value=5, value=1)

        if st.session_state.start_node and st.session_state.goal_node:
            if st.button("Find Path"):
                st.session_state.path = bms_algorithm(st.session_state.graph, st.session_state.start_node, st.session_state.goal_node, st.session_state.max_goals)
                if st.session_state.path:
                    st.success(f"Paths found: {st.session_state.path}")
                else:
                    st.error("No path found.")

    if st.session_state.path or st.session_state.goal_node:
        st.subheader("Full Graph:")
        visualize_graph(st.session_state.graph)

        if st.session_state.path:
            st.subheader("Graph with Paths:")
            for path in st.session_state.path:
                visualize_graph(st.session_state.graph, path)

    if st.button("Reset"):
        st.session_state.clear()  

# BMS Algorithm Implementation
def bms_algorithm(graph, start, goal, max_goal_cutoff):
    queue = [[start]]
    res = []
    goal_found = 0

    while queue and goal_found < max_goal_cutoff:
        curr_path = queue.pop(0)
        curr_node = curr_path[-1]

        neighbors = sorted(graph.neighbors(curr_node))  
        for neighbor in neighbors:
            if neighbor not in curr_path:
                new_path = list(curr_path)
                new_path.append(neighbor)
                queue.append(new_path)

                if neighbor == goal:
                    res.append(new_path)
                    goal_found += 1
                    if goal_found == max_goal_cutoff:
                        return res

    return res

bms_search()
