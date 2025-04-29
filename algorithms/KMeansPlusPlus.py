from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

def run_kmeans_plusplus(X_scaled, n_clusters, scaler, features):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    centroids = kmeans.cluster_centers_

    # Return both result dictionary and labels
    return {
        'centroids': centroids,
        'labels': labels
    }, labels
