import pandas as pd
import networkx as nx
import pickle
from tabulate import tabulate


# Function to build or load the graph
def build_or_load_graph():
    try:
        with open('undirected_weighted_graph.pkl', 'rb') as file:
            G = pickle.load(file)
        print("Graph loaded from file.")
    except FileNotFoundError:
        G = build_graph()
        with open('undirected_weighted_graph.pkl', 'wb') as file:
            pickle.dump(G, file)
        print("Graph built and saved to file.")
    return G


# Function to build the undirected graph
def build_graph():
    # Load the dataset with proper handling of mixed types
    df = pd.read_csv('../dataset/dataset.csv', low_memory=False)
    # Create an undirected graph
    G = nx.Graph()
    # Add channel nodes with 'node_type' attribute set to 'channel'
    for index, row1 in df.iterrows():
        G.add_node(row1['channelId'], title=row1['channelTitle'], node_type='channel')
    # Add video nodes with 'node_type' attribute set to 'video'
    videos = set(df['video_id'])
    G.add_nodes_from(videos, node_type='video')

    # Add edges with weights based on views, comments, likes, and dislikes
    for index, row in df.iterrows():
        channel_id = row['channelId']
        video_id = row['video_id']
        views = row['view_count']
        comments = row['comment_count']
        likes = row['likes']
        dislikes = row['dislikes']

        weight = views + 4 * comments + 2 * likes - 2 * dislikes

        G.add_edge(channel_id, video_id, weight=weight)

    return G
def display_results(top_channels, centrality_dict, G):
    table_data = []

    for channel in top_channels:
        # Skip invalid or unexpected values in 'channelId'
        if pd.notna(channel):
            title = G.nodes[channel].get('title', 'Title not available')
            degree = G.degree(channel)
            w_degree = centrality_dict[channel]
            w_eigenvector = weighted_eigenvector_centrality[channel]
            w_pagerank = weighted_pagerank[channel]

            table_data.append([channel, title, degree, w_degree, w_eigenvector, w_pagerank])

    headers = ["Channel ID", "Title", "Degree", "Weighted Degree", "Weighted Eigenvector", "Weighted PageRank"]
    table = tabulate(table_data, headers=headers, tablefmt="pretty")

    print(f"\nTop {len(top_channels)} channels:")
    print(table)

# Build or load the undirected graph
G = build_or_load_graph()

degree_centrality = nx.degree_centrality(G)
weighted_degree_centrality = dict(G.degree(weight='weight'))
weighted_eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
weighted_pagerank = nx.pagerank(G, weight='weight')

# Print top 15 channels based on activity and showing impression
top_channels_degree = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:15]
display_results(top_channels_degree, weighted_degree_centrality, G)