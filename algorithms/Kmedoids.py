import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import List, Tuple, Dict
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class KMedoidsParameters:
    n_clusters: int
    max_iter: int = 300
    tol: float = 1e-4
    init_method: str = 'random'  # 'random' or 'k-means++'
    random_state: int = 42
    metric: str = 'euclidean'  # 'euclidean' or 'manhattan'

class KMedoidsClustering:
    def __init__(self, params: KMedoidsParameters):
        self.params = params
        if self.params.n_clusters <= 0:
            raise ValueError("Number of clusters must be positive")
        np.random.seed(params.random_state)
        
    def calculate_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances between all points"""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        if self.params.metric == 'euclidean':
            for i in range(n_samples):
                distances[i] = np.sqrt(np.sum((X - X[i])**2, axis=1))
        elif self.params.metric == 'manhattan':
            for i in range(n_samples):
                distances[i] = np.sum(np.abs(X - X[i]), axis=1)
        else:
            raise ValueError("Invalid metric. Choose 'euclidean' or 'manhattan'")
                
        return distances
    
    def initialize_medoids(self, X: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """Initialize medoids using specified method"""
        n_samples = X.shape[0]
        
        if self.params.n_clusters >= n_samples:
            raise ValueError("Number of clusters must be less than number of samples")
            
        if self.params.init_method == 'random':
            medoid_indices = np.random.choice(
                n_samples, 
                self.params.n_clusters, 
                replace=False
            )
        else:  # k-means++
            medoid_indices = [np.random.randint(n_samples)]
            
            for _ in range(self.params.n_clusters - 1):
                dist_to_medoids = np.min(distances[medoid_indices], axis=0)
                probabilities = dist_to_medoids**2 / np.sum(dist_to_medoids**2)
                medoid_indices.append(np.random.choice(n_samples, p=probabilities))
                
        return np.array(medoid_indices)
    
    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit K-Medoids to the data"""
        distances = self.calculate_distances(X)
        medoid_indices = self.initialize_medoids(X, distances)
        prev_cost = float('inf')
        
        with tqdm(total=self.params.max_iter, desc="K-Medoids Progress") as pbar:
            for _ in range(self.params.max_iter):
                # Assign points to nearest medoids
                labels = np.argmin(distances[medoid_indices], axis=0)
                
                # Update medoids
                new_medoid_indices = medoid_indices.copy()
                current_cost = np.sum(np.min(distances[medoid_indices], axis=0))
                
                for i in range(self.params.n_clusters):
                    cluster_points = np.where(labels == i)[0]
                    if len(cluster_points) > 0:
                        costs = np.sum(distances[cluster_points][:, cluster_points], axis=1)
                        best_medoid = cluster_points[np.argmin(costs)]
                        new_medoid_indices[i] = best_medoid
                
                if np.array_equal(medoid_indices, new_medoid_indices):
                    pbar.set_postfix({'status': 'converged'})
                    break
                    
                medoid_indices = new_medoid_indices
                
                if abs(prev_cost - current_cost) < self.params.tol:
                    pbar.set_postfix({'status': 'converged (tol)'})
                    break
                prev_cost = current_cost
                pbar.update(1)
                
        return medoid_indices, labels

def run_kmedoids_clustering(X: np.ndarray, n_clusters: int) -> Tuple[List[Dict], List[np.ndarray]]:
    """Run K-Medoids with 4 different parameter settings"""
    
    parameter_versions = {
        'fast': KMedoidsParameters(
            n_clusters=n_clusters,
            max_iter=100,
            tol=1e-3,
            init_method='random',
            metric='euclidean'
        ),
        'balanced': KMedoidsParameters(
            n_clusters=n_clusters,
            max_iter=300,
            tol=1e-4,
            init_method='k-means++',
            metric='euclidean'
        ),
        'thorough': KMedoidsParameters(
            n_clusters=n_clusters,
            max_iter=500,
            tol=1e-5,
            init_method='k-means++',
            metric='manhattan'
        ),
        'robust': KMedoidsParameters(
            n_clusters=n_clusters,
            max_iter=400,
            tol=1e-4,
            init_method='k-means++',
            metric='manhattan'
        )
    }
    
    all_results = []
    all_labels = []
    
    for version_name, params in parameter_versions.items():
        try:
            kmedoids = KMedoidsClustering(params)
            medoid_indices, labels = kmedoids.fit(X)
            
            # Calculate metrics only if we have at least 2 clusters
            unique_clusters = len(np.unique(labels))
            metrics = {}
            
            if unique_clusters > 1:
                metrics = {
                    'silhouette': silhouette_score(X, labels),
                    'Davies-Bouldin': davies_bouldin_score(X, labels),
                    'Calinski-Harabasz': calinski_harabasz_score(X, labels)
                }
            else:
                metrics = {
                    'silhouette': None,
                    'Davies-Bouldin': None,
                    'Calinski-Harabasz': None
                }
            
            results = {
                'method': f'KMedoids_{version_name}',
                'parameters': {
                    'max_iter': params.max_iter,
                    'tol': params.tol,
                    'init_method': params.init_method,
                    'metric': params.metric
                },
                'medoid_indices': medoid_indices,
                'metrics': metrics,
                'n_clusters': unique_clusters
            }
            
            all_results.append(results)
            all_labels.append(labels)
            
        except Exception as e:
            print(f"Error running K-Medoids {version_name}: {str(e)}")
            continue
    
    return all_results, all_labels