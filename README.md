Using API.ipynb:
This code interacts with the YouTube API to gather channel and video data. It searches for channels based on specific keywords and retrieves statistics like views, subscribers, and video count. For each channel, the code gathers video details such as titles, descriptions, and publication dates, along with user comments for each video. It then analyzes the text from these comments, filters out common stop words, and identifies the most frequently used words in the comments. due to the limited information provided by the free API, I used a prepared dataset for the continuation.

Step 2:
This code builds or loads an undirected graph from a dataset of videos, where nodes represent videos and edges connect videos with shared tags. It then calculates and prints the top 5 nodes based on various centrality measures (Degree, Betweenness, Closeness, Eigenvector, PageRank) and visualizes the graph.
Step 3:
In the first part, The code analyzes sentiment from video titles, descriptions, and tags in a CSV file. It tokenizes and cleans the text, then uses VADER to generate sentiment scores for positive, negative, neutral, and overall sentiment. 
In the second part, the code extracts the top 25 keywords from a dataset's "tags" column, removes stop words, and performs sentiment analysis to classify the keywords as positive, negative, or neutral.

Step 4:
The code creates or loads an undirected weighted graph of YouTube channels and videos. It adds nodes for channels and videos, calculates weights based on views, comments, likes, and dislikes, computes centrality metrics, and displays the top channels in a formatted table.

Step 5:
The code aims to identify and analyze communities within a graph of YouTube videos and channels. It detects communities, preprocesses tags, applies K-Means clustering to categorize videos, and visualizes the community structure to understand shared topics among videos.
