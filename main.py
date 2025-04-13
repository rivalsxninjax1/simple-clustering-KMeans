# clustering.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# Step 1: Read the data
data = pd.read_csv('data.csv')
print("Data Head:")
print(data.head())

# Step 2: Pre-process the Data
# We will use 'AnnualIncome' and 'SpendingScore' for clustering.
X = data[['AnnualIncome', 'SpendingScore']]

# Step 3: Determine the Number of Clusters
# You might use the elbow method to decide the optimal number of clusters.
# In this simple example, we choose 3 clusters.
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Step 4: Add the Cluster Labels back to the DataFrame
data['Cluster'] = kmeans.labels_
print("\nData with Cluster Labels:")
print(data.head())

# Step 5: Visualize the Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='viridis', s=100)

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.xlabel("Annual Income (K)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments based on Annual Income and Spending Score")
plt.legend()
plt.grid(True)
plt.show()
