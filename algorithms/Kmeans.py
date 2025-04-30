# change
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class KMeansParameters:
    n_clusters: int
    n_init: int = 10
    max_iter: int = 300
    init: str = 'k-means++'
    random_state: int = 42
    algorithm: str = 'lloyd'

def run_kmeans_single(X: np.ndarray, params: KMeansParameters) -> Dict:
    """Run KMeans with specific parameters"""
    kmeans = KMeans(
        n_clusters=params.n_clusters,
        n_init=params.n_init,
        max_iter=params.max_iter,
        init=params.init,
        random_state=params.random_state,
        algorithm=params.algorithm
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

def run_kmeans_clustering(X: np.ndarray, n_clusters: int) -> Tuple[List[Dict], List[np.ndarray]]:
    """Run KMeans with 4 different parameter settings"""
    
    parameter_versions = {
        'quick': KMeansParameters(
            n_clusters=n_clusters,
            n_init=5,
            max_iter=200,
            init='random',
            algorithm='lloyd'
        ),
        'balanced': KMeansParameters(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            init='k-means++',
            algorithm='lloyd'
        ),
        'thorough': KMeansParameters(
            n_clusters=n_clusters,
            n_init=20,
            max_iter=500,
            init='k-means++',
            algorithm='full'
        ),
        'elkan': KMeansParameters(
            n_clusters=n_clusters,
            n_init=15,
            max_iter=400,
            init='k-means++',
            algorithm='elkan'
        )
    }
    
    all_results = []
    all_labels = []
    
    for version_name, params in parameter_versions.items():
        result = run_kmeans_single(X, params)
        
        results = {
            'method': f'KMeans_{version_name}',
            'parameters': params,
            'centroids': result['centroids'],
            'metrics': result['metrics'],
            'n_iterations': result['n_iterations']
        }
        
        all_results.append(results)
        all_labels.append(result['labels'])
    
    return all_results, all_labels

def plot_kmeans_comparison(X: np.ndarray, results: List[Dict], labels: List[np.ndarray]):
    """Plot clustering results for all versions"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(20, 5))
    for idx, (result, cluster_labels) in enumerate(zip(results, labels)):
        plt.subplot(1, 4, idx + 1)
        
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            mask = cluster_labels == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {label}', alpha=0.6)
            
        centroids_pca = pca.transform(result['centroids'])
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                c='red', marker='x', s=200, linewidth=3, label='Centroids')
        
        plt.title(f"{result['method']}\nSilhouette: {result['metrics']['silhouette']:.3f}")
        if idx == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.show()

def print_comparison_results(results: List[Dict]):
    """Print comparison metrics for all versions"""
    print("\nKMeans Versions Comparison:")
    print("-" * 60)
    print(f"{'Version':<15} {'Silhouette':<12} {'DB Index':<12} {'CH Score':<12} {'Iterations':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['method']:<15} "
            f"{result['metrics']['silhouette']:<12.3f} "
            f"{result['metrics']['davies_bouldin']:<12.3f} "
            f"{result['metrics']['calinski_harabasz']:<12.0f} "
            f"{result['n_iterations']:<10}")