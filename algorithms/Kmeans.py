import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

def find_optimal_k(X_scaled, k_min=2, k_max=10):
    """
    Use Elbow method and Silhouette scores to find optimal number of clusters.
    """
    inertia = []
    silhouette_scores = []
    k_range = range(k_min, k_max + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')

    plt.tight_layout()
    plt.show()
    plt.close()


def apply_kmeans(X_scaled, optimal_k=6):
    """
    Apply K-means clustering.
    """
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_centroids = kmeans.cluster_centers_
    return kmeans_labels, kmeans_centroids


def evaluate_kmeans(X_scaled, kmeans_labels):
    """
    Evaluate K-means clustering performance.
    """
    silhouette = silhouette_score(X_scaled, kmeans_labels)
    db_index = davies_bouldin_score(X_scaled, kmeans_labels)
    ch_score = calinski_harabasz_score(X_scaled, kmeans_labels)

    print("\nK-means Clustering Evaluation:")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print(f"Calinski-Harabasz Index: {ch_score:.4f}")

    return silhouette, db_index, ch_score


def plot_pca_clusters(X_scaled, kmeans_labels, kmeans_centroids, optimal_k=6):
    """
    Plot the K-means clustering result after applying PCA.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    for i in range(optimal_k):
        plt.scatter(X_pca[kmeans_labels == i, 0], X_pca[kmeans_labels == i, 1], label=f'Cluster {i+1}')

    centroids_pca = pca.transform(kmeans_centroids)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=100, c='black', marker='X', label='Centroids')

    plt.title('K-means Clustering Results (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
    plt.close()


def plot_original_features(df, kmeans_centroids, scaler):
    """
    Plot clustering result using original features like Annual Income and Spending Score.
    """
    optimal_k = len(np.unique(df['KMeans_Cluster']))

    plt.figure(figsize=(10, 8))
    for i in range(optimal_k):
        cluster_data = df[df['KMeans_Cluster'] == i]
        plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {i+1}')

    centroids_original = scaler.inverse_transform(kmeans_centroids)
    plt.scatter(centroids_original[:, 1], centroids_original[:, 2], s=100, c='black', marker='X', label='Centroids')

    plt.title('K-means Clustering Results (Income vs Spending)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    plt.close()
