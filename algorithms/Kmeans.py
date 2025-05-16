import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_kmeans_clustering(X, n_clusters):
    results_list = []
    labels_list = []
    
    # Version 1: Standard K-Means
    kmeans_standard = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )
    
    labels_standard = kmeans_standard.fit_predict(X)
    
    # Calculate metrics
    if len(np.unique(labels_standard)) > 1:
        sil_score_standard = silhouette_score(X, labels_standard)
        db_score_standard = davies_bouldin_score(X, labels_standard)
        ch_score_standard = calinski_harabasz_score(X, labels_standard)
    else:
        sil_score_standard = 0
        db_score_standard = float('inf')
        ch_score_standard = 0
    
    results_standard = {
        'method': 'KMeans_Standard',
        'centroids': kmeans_standard.cluster_centers_,
        'inertia': kmeans_standard.inertia_,
        'n_iter': kmeans_standard.n_iter_,
        'metrics': {
            'silhouette': sil_score_standard,
            'Davies-Bouldin': db_score_standard,
            'Calinski-Harabasz': ch_score_standard
        }
    }
    
    results_list.append(results_standard)
    labels_list.append(labels_standard)
    
    # Version 2: Hierarchical Clustering (Agglomerative)
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    
    labels_hierarchical = hierarchical.fit_predict(X)
    
    # Calculate metrics
    if len(np.unique(labels_hierarchical)) > 1:
        sil_score_hierarchical = silhouette_score(X, labels_hierarchical)
        db_score_hierarchical = davies_bouldin_score(X, labels_hierarchical)
        ch_score_hierarchical = calinski_harabasz_score(X, labels_hierarchical)
    else:
        sil_score_hierarchical = 0
        db_score_hierarchical = float('inf')
        ch_score_hierarchical = 0
    
    results_hierarchical = {
        'method': 'Hierarchical',
        'centroids': None,  # Hierarchical doesn't provide centroids by default
        'inertia': None,    # Inertia not applicable
        'n_iter': None,     # No iterations
        'metrics': {
            'silhouette': sil_score_hierarchical,
            'Davies-Bouldin': db_score_hierarchical,
            'Calinski-Harabasz': ch_score_hierarchical
        }
    }
    
    results_list.append(results_hierarchical)
    labels_list.append(labels_hierarchical)
    
    # Version 3: Bisecting K-Means
    # Implementing bisecting k-means manually
    def bisecting_kmeans(X, n_clusters, max_iter=300, random_state=42):
        np.random.seed(random_state)
        labels = np.zeros(X.shape[0], dtype=int)
        cluster_centers = [np.mean(X, axis=0)]
        cluster_sizes = [X.shape[0]]
        
        while len(cluster_centers) < n_clusters:
            # Select cluster with highest SSE (inertia)
            max_sse = -1
            split_idx = -1
            for i in range(len(cluster_centers)):
                cluster_points = X[labels == i]
                if len(cluster_points) < 2:
                    continue
                sse = np.sum((cluster_points - cluster_centers[i])**2)
                if sse > max_sse:
                    max_sse = sse
                    split_idx = i
            
            if split_idx == -1:
                break
                
            # Split the selected cluster
            cluster_points = X[labels == split_idx]
            kmeans_split = KMeans(
                n_clusters=2,
                init='k-means++',
                max_iter=max_iter,
                n_init=10,
                random_state=random_state
            )
            sub_labels = kmeans_split.fit_predict(cluster_points)
            
            # Update labels
            new_labels = np.zeros_like(labels)
            new_centers = []
            new_sizes = []
            new_cluster_id = len(cluster_centers)
            
            for i in range(len(cluster_centers)):
                if i == split_idx:
                    # Split cluster
                    points = X[labels == i]
                    sub_centers = kmeans_split.cluster_centers_
                    new_labels[labels == i] = np.where(sub_labels == 0, i, new_cluster_id)
                    new_centers.append(sub_centers[0])
                    new_centers.append(sub_centers[1])
                    new_sizes.append(np.sum(sub_labels == 0))
                    new_sizes.append(np.sum(sub_labels == 1))
                else:
                    # Keep existing cluster
                    new_labels[labels == i] = i
                    new_centers.append(cluster_centers[i])
                    new_sizes.append(cluster_sizes[i])
            
            labels = new_labels
            cluster_centers = new_centers
            cluster_sizes = new_sizes
        
        inertia = 0
        for i in range(len(cluster_centers)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - cluster_centers[i])**2)
        
        return labels, np.array(cluster_centers), inertia
    
    labels_bisecting, centers_bisecting, inertia_bisecting = bisecting_kmeans(X, n_clusters)
    
    # Calculate metrics
    if len(np.unique(labels_bisecting)) > 1:
        sil_score_bisecting = silhouette_score(X, labels_bisecting)
        db_score_bisecting = davies_bouldin_score(X, labels_bisecting)
        ch_score_bisecting = calinski_harabasz_score(X, labels_bisecting)
    else:
        sil_score_bisecting = 0
        db_score_bisecting = float('inf')
        ch_score_bisecting = 0
    
    results_bisecting = {
        'method': 'Bisecting_KMeans',
        'centroids': centers_bisecting,
        'inertia': inertia_bisecting,
        'n_iter': None,  # Not tracked in this implementation
        'metrics': {
            'silhouette': sil_score_bisecting,
            'Davies-Bouldin': db_score_bisecting,
            'Calinski-Harabasz': ch_score_bisecting
        }
    }
    
    results_list.append(results_bisecting)
    labels_list.append(labels_bisecting)
    
    return results_list, labels_list

def calculate_optimal_clusters(X, max_clusters=10):
    # Calculate distortion (inertia) for a range of number of clusters
    K = range(1, min(max_clusters+1, X.shape[0]))
    inertias = []
    silhouette_scores = []
    
    for k in K:
        if k == 1:
            # For k=1, just calculate inertia
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(0)  # Silhouette score not defined for k=1
        else:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            try:
                silhouette_scores.append(silhouette_score(X, labels))
            except:
                silhouette_scores.append(0)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot elbow curve (inertia)
    ax1.plot(K, inertias, 'bo-')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Plot silhouette scores
    ax2.plot(K[1:], silhouette_scores[1:], 'ro-')  # Skip k=1 for silhouette
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for Optimal k')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Determine optimal k using the elbow method
    inertia_diffs = np.diff(inertias)
    inertia_diffs_rate = np.diff(inertia_diffs)
    
    if len(inertia_diffs_rate) > 0:
        elbow_index = np.argmax(inertia_diffs_rate) + 1
        optimal_k = K[elbow_index]
    else:
        optimal_k = 2  # Default if we can't determine
    
    if any(s > 0 for s in silhouette_scores[1:]):
        sil_optimal_k = K[1:][np.argmax(silhouette_scores[1:])]
    else:
        sil_optimal_k = optimal_k
    
    final_optimal_k = (optimal_k + sil_optimal_k) // 2
    
    # Add a vertical line indicating the optimal k
    ax1.axvline(x=final_optimal_k, color='r', linestyle='--')
    ax2.axvline(x=final_optimal_k, color='r', linestyle='--')
    
    ax1.text(final_optimal_k, max(inertias)*0.9, f'Optimal k = {final_optimal_k}', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    return fig, final_optimal_k