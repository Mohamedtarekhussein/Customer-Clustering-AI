# algorithms/AgglomerativeClustering/agglomerative_clustering.py

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

def run_agglomerative_clustering(X_scaled, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    return {
        'labels': labels,
        'centroids': np.zeros((n_clusters, X_scaled.shape[1])),  # dummy centroids
        'metrics': {
            'silhouette': silhouette,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score
        }
    }
