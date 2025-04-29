# algorithms/DBSCAN/dbscan_clustering.py

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

def run_dbscan_clustering(X_scaled, df):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    df['DBSCAN_Cluster'] = dbscan_labels

    mask = dbscan_labels != -1

    if len(set(dbscan_labels[mask])) > 1:
        silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
        db_score = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
        ch_score = calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])

        return {
            'labels': dbscan_labels,
            'centroids': np.zeros((n_clusters_dbscan, X_scaled.shape[1])),
            'metrics': {
                'silhouette': silhouette,
                'davies_bouldin': db_score,
                'calinski_harabasz': ch_score
            }
        }
    else:
        return {
            'labels': dbscan_labels,
            'centroids': None,
            'metrics': {
                'error': 'DBSCAN found only one cluster or noise.'
            }
        }
