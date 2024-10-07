import pickle
from networkx.algorithms import community
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re


def load_graph():
    try:
        with open('undirected_graph.pkl', 'rb') as file:
            G = pickle.load(file)
        print("Graph loaded from file.")
    except FileNotFoundError:
        print("Graph loading failed.")
    return G


def clean_and_preprocess(tag):
    tag = tag.lower()
    tag = re.sub(r'[^a-zA-Z0-9\s]', '', tag)
    return tag


G = load_graph()
# Perform community detection using the Louvain method
communities = list(community.greedy_modularity_communities(G))

for i, community in enumerate(communities):
    titles = [G.nodes[video_id].get('title', f'Title not available for {video_id}') for video_id in community]
    print(f"Community {i + 1}: {titles}")

color_map = {}
for i, community in enumerate(communities):
    for video_id in community:
        color_map[video_id] = i

# Draw the graph with nodes colored by community
pos = nx.spring_layout(G, seed=42)  # Using seed for reproducibility
plt.figure(figsize=(16, 12))
nx.draw(G, pos, node_color=[color_map[node] for node in G.nodes], with_labels=False, node_size=20, cmap='viridis', alpha=0.7)
plt.title('Community Detection Visualization')
plt.show()

print('***************************************************')

df = pd.read_csv('../dataset/dataset.csv', low_memory=False)
community_df = df[df['video_id'].isin(communities[0])].copy()  # Create a copy of the DataFrame

tags_list = community_df['tags'].dropna().tolist()

preprocessed_tags = [clean_and_preprocess(tag) for tag in tags_list]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tags_vectors = vectorizer.fit_transform(preprocessed_tags)

# K-Means Clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(tags_vectors)

# Create a new DataFrame with the 'cluster' information
result_df = community_df.copy()
result_df['cluster'] = cluster_assignments


for cluster_id in range(num_clusters):
    cluster_videos = result_df[result_df['cluster'] == cluster_id]['title']
    print(f"Cluster {cluster_id+1} Videos:")
    print(cluster_videos)
    print("\n")
