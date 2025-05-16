from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class DBSCANParameters:
    eps: float
    min_samples: int
    metric: str = 'euclidean'
    algorithm: str = 'auto'
    leaf_size: int = 30
    n_jobs: int = -1

def run_dbscan_single(X: np.ndarray, params: DBSCANParameters) -> Dict:
    """Run DBSCAN with specific parameters"""
    dbscan = DBSCAN(
        eps=params.eps,
        min_samples=params.min_samples,
        metric=params.metric,
        algorithm=params.algorithm,
        leaf_size=params.leaf_size,
        n_jobs=params.n_jobs
    )
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    mask = labels != -1
    
    if len(set(labels[mask])) > 1:
        metrics = {
            'silhouette': silhouette_score(X[mask], labels[mask]),
            'Davies-Bouldin': davies_bouldin_score(X[mask], labels[mask]),
            'Calinski-Harabasz': calinski_harabasz_score(X[mask], labels[mask])
        }
    else:
        metrics = {
            'silhouette': None,
            'Davies-Bouldin': None,
            'Calinski-Harabasz': None
        }
    
    return {
        'method': f"DBSCAN (eps={params.eps}, min_samples={params.min_samples})",
        'labels': labels,
        'n_clusters': n_clusters,
        'metrics': metrics,
        'parameters': {
            'eps': params.eps,
            'min_samples': params.min_samples
        }
    }

def run_dbscan_clustering(X: np.ndarray) -> Tuple[List[Dict], List[np.ndarray]]:
    """Run DBSCAN with 4 different parameter settings"""
    
    # Define 4 parameter versions
    parameter_versions = {
        'conservative': DBSCANParameters(
            eps=0.5,
            min_samples=5,
            metric='euclidean',
            algorithm='auto'
        ),
        'aggressive': DBSCANParameters(
            eps=0.3,
            min_samples=3,
            metric='euclidean',
            algorithm='auto'
        ),
        'balanced': DBSCANParameters(
            eps=0.4,
            min_samples=4,
            metric='euclidean',
            algorithm='auto'
        ),
        'dense': DBSCANParameters(
            eps=0.2,
            min_samples=6,
            metric='euclidean',
            algorithm='auto'
        )
    }
    
    all_results = []
    all_labels = []
    
    for version_name, params in parameter_versions.items():
        result = run_dbscan_single(X, params)
        all_results.append(result)
        all_labels.append(result['labels'])
    
    return all_results, all_labels