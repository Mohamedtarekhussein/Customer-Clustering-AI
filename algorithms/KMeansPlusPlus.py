# change
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    pairwise_distances_argmin_min
)
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class KMeansPlusPlusParameters:
    n_clusters: int
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42

def run_kmeans_plusplus_single(X: np.ndarray, params: KMeansPlusPlusParameters) -> Dict:
    """Run K-means++ with specific parameters"""
    kmeans = KMeans(
        n_clusters=params.n_clusters,
        init='k-means++',
        n_init=params.n_init,
        max_iter=params.max_iter,
        tol=params.tol,
        random_state=params.random_state
    )
    
    labels = kmeans.fit_predict(X)
    
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'inertia': kmeans.inertia_
    }
    
    return {
        'labels': labels,
        'centroids': kmeans.cluster_centers_,
        'metrics': metrics,
        'n_iterations': kmeans.n_iter_
    }

def run_kmeans_plusplus(X: np.ndarray, n_clusters: int) -> Tuple[List[Dict], List[np.ndarray]]:
    """Run K-means++ with 4 different parameter settings"""
    
    parameter_versions = {
        'fast': KMeansPlusPlusParameters(
            n_clusters=n_clusters,
            n_init=5,
            max_iter=200,
            tol=1e-3
        ),
        'balanced': KMeansPlusPlusParameters(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            tol=1e-4
        ),
        'thorough': KMeansPlusPlusParameters(
            n_clusters=n_clusters,
            n_init=20,
            max_iter=500,
            tol=1e-5
        ),
        'precise': KMeansPlusPlusParameters(
            n_clusters=n_clusters,
            n_init=15,
            max_iter=400,
            tol=1e-6
        )
    }
    
    all_results = []
    all_labels = []
    
    for version_name, params in parameter_versions.items():
        result = run_kmeans_plusplus_single(X, params)
        
        results = {
            'method': f'KMeans++_{version_name}',
            'parameters': params,
            'centroids': result['centroids'],
            'metrics': result['metrics'],
            'n_iterations': result['n_iterations']
        }
        
        all_results.append(results)
        all_labels.append(result['labels'])
    
    return all_results, all_labels

def print_kmeanspp_comparison(results: List[Dict]):
    """Print comparison metrics for all versions"""
    print("\nK-means++ Versions Comparison:")
    print("-" * 70)
    print(f"{'Version':<15} {'Silhouette':<12} {'DB Index':<12} {'CH Score':<12} {'Iterations':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['method']:<15} "
            f"{result['metrics']['silhouette']:<12.3f} "
            f"{result['metrics']['davies_bouldin']:<12.3f} "
            f"{result['metrics']['calinski_harabasz']:<12.0f} "
            f"{result['n_iterations']:<10}")