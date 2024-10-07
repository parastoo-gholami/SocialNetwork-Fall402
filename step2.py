import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# Function to build or load the graph
def build_or_load_graph():
    try:
        with open('undirected_graph.pkl', 'rb') as file:
            G = pickle.load(file)
        print("Graph loaded from file.")
    except FileNotFoundError:
        G = build_graph()
        with open('undirected_graph.pkl', 'wb') as file:
            pickle.dump(G, file)
        print("Graph built and saved to file.")
    return G

# Function to build the undirected graph
def build_graph():
    # Load the dataset with proper handling of mixed types
    df = pd.read_csv('../dataset/dataset.csv', low_memory=False)

    # Create an undirected graph
    G = nx.Graph()

    # Add videos as nodes and connect them based on shared tags
    for index, row1 in df.iterrows():
        if isinstance(row1['tags'], str):  # Check for NaN or non-string values in tags
            G.add_node(row1['video_id'], title=row1['title'], tags=row1['tags'])
            for index, row2 in df.iterrows():
                if (
                    row1['video_id'] != row2['video_id']
                    and isinstance(row2['tags'], str)  # Check for NaN or non-string values in tags
                    and set(row1['tags'].split('|')).intersection(row2['tags'].split('|'))
                ):
                    G.add_edge(row1['video_id'], row2['video_id'])

    return G

# Build or load the undirected graph
undirected_graph = build_or_load_graph()

print(f"Number of nodes: {len(undirected_graph.nodes)}")
print(f"Number of edges: {len(undirected_graph.edges)}")

# Filter out nodes with 'nan' values
valid_nodes = [node for node in undirected_graph.nodes if isinstance(node, str)]

# Degree Centrality
degree_centrality = nx.degree_centrality(undirected_graph.subgraph(valid_nodes))
top_nodes_degree = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:5]
print("Top 5 nodes by Degree Centrality:")
for node in top_nodes_degree:
    print(f"Node: {node}, Title: {undirected_graph.nodes[node]['title']}, Degree Centrality: {degree_centrality[node]}")

# Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(undirected_graph.subgraph(valid_nodes))
top_nodes_betweenness = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:5]
print("\nTop 5 nodes by Betweenness Centrality:")
for node in top_nodes_betweenness:
    print(f"Node: {node}, Title: {undirected_graph.nodes[node]['title']}, Betweenness Centrality: {betweenness_centrality[node]}")

# Closeness Centrality
closeness_centrality = nx.closeness_centrality(undirected_graph.subgraph(valid_nodes))
top_nodes_closeness = sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)[:5]
print("\nTop 5 nodes by Closeness Centrality:")
for node in top_nodes_closeness:
    print(f"Node: {node}, Title: {undirected_graph.nodes[node]['title']}, Closeness Centrality: {closeness_centrality[node]}")

# Eigenvector Centrality
eigenvector_centrality = nx.eigenvector_centrality(undirected_graph.subgraph(valid_nodes))
top_nodes_eigenvector = sorted(eigenvector_centrality, key=eigenvector_centrality.get, reverse=True)[:5]
print("\nTop 5 nodes by Eigenvector Centrality:")
for node in top_nodes_eigenvector:
    print(f"Node: {node}, Title: {undirected_graph.nodes[node]['title']}, Eigenvector Centrality: {eigenvector_centrality[node]}")

# PageRank
pagerank = nx.pagerank(undirected_graph.subgraph(valid_nodes))
top_nodes_pagerank = sorted(pagerank, key=pagerank.get, reverse=True)[:5]
print("\nTop 5 nodes by PageRank:")
for node in top_nodes_pagerank:
    print(f"Node: {node}, Title: {undirected_graph.nodes[node]['title']}, PageRank: {pagerank[node]}")

# Plot the graph
pos = nx.spring_layout(undirected_graph)
nx.draw(undirected_graph, pos, with_labels=False, node_size=10, alpha=0.6)
plt.show()
