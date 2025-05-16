import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import List, Tuple, Dict
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class FCMParameters:
    n_clusters: int
    max_iter: int = 200
    m: float = 2.0  # fuzziness parameter
    error: float = 1e-5
    random_state: int = 42
    init_method: str = 'random'  # 'random' or 'kmeans++'

class FuzzyCMeans:
    def __init__(self, params: FCMParameters):
        self.params = params
        if self.params.m <= 1:
            raise ValueError("Fuzziness parameter m must be greater than 1")
        if self.params.n_clusters <= 0:
            raise ValueError("Number of clusters must be positive")
        np.random.seed(params.random_state)
        
    def initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centroids"""
        n_samples, n_features = X.shape
        
        if self.params.n_clusters >= n_samples:
            raise ValueError("Number of clusters must be less than number of samples")
            
        if self.params.init_method == 'random':
            idx = np.random.choice(n_samples, self.params.n_clusters, replace=False)
            return X[idx]
        else:  # kmeans++
            centroids = [X[np.random.randint(n_samples)]]
            
            for _ in range(self.params.n_clusters - 1):
                dist = np.array([min([np.inner(c-x, c-x) for c in centroids]) 
                              for x in X])
                probs = dist / dist.sum()
                cumprobs = probs.cumsum()
                r = np.random.random()
                
                for j, p in enumerate(cumprobs):
                    if r < p:
                        centroids.append(X[j])
                        break
            
            return np.array(centroids)
    
    def update_membership(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Update membership matrix"""
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        membership = np.zeros((n_samples, n_clusters))
        
        for i in range(n_samples):
            distances = np.linalg.norm(centroids - X[i], axis=1)
            if np.any(distances == 0):
                membership[i] = np.where(distances == 0, 1, 0)
            else:
                distances = distances**(2/(self.params.m-1))
                membership[i] = 1 / np.sum(
                    (distances.reshape(-1, 1) / distances)**2, axis=0
                )
        
        # Ensure rows sum to 1
        membership = membership / (membership.sum(axis=1, keepdims=True) + 1e-10)
        return membership
    
    def update_centroids(self, X: np.ndarray, membership: np.ndarray) -> np.ndarray:
        """Update cluster centroids"""
        powered_membership = membership ** self.params.m
        return (powered_membership.T @ X) / (powered_membership.sum(axis=0)[:, None] + 1e-10)
    
    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit Fuzzy C-Means to the data"""
        centroids = self.initialize_centroids(X)
        
        with tqdm(total=self.params.max_iter, desc="FCM Progress") as pbar:
            for _ in range(self.params.max_iter):
                old_centroids = centroids.copy()
                membership = self.update_membership(X, centroids)
                centroids = self.update_centroids(X, membership)
                
                if np.linalg.norm(centroids - old_centroids) < self.params.error:
                    pbar.set_postfix({'status': 'converged'})
                    break
                pbar.update(1)
                
        labels = np.argmax(membership, axis=1)
        return centroids, labels, membership

def run_fcm_clustering(X: np.ndarray, n_clusters: int) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray]]:
    """Run FCM with 4 different parameter settings"""
    
    parameter_versions = {
        'quick': FCMParameters(
            n_clusters=n_clusters,
            max_iter=100,
            m=2.0,
            error=1e-4,
            init_method='random'
        ),
        'balanced': FCMParameters(
            n_clusters=n_clusters,
            max_iter=200,
            m=2.0,
            error=1e-5,
            init_method='kmeans++'
        ),
        'fine': FCMParameters(
            n_clusters=n_clusters,
            max_iter=300,
            m=1.8,
            error=1e-6,
            init_method='kmeans++'
        ),
        'fuzzy': FCMParameters(
            n_clusters=n_clusters,
            max_iter=250,
            m=2.5,
            error=1e-5,
            init_method='kmeans++'
        )
    }
    
    all_results = []
    all_labels = []
    all_memberships = []
    
    for version_name, params in parameter_versions.items():
        try:
            fcm = FuzzyCMeans(params)
            centroids, labels, membership = fcm.fit(X)
            
            # Calculate metrics only if we have at least 2 clusters
            unique_clusters = len(np.unique(labels))
            metrics = {}
            
            if unique_clusters > 1:
                metrics = {
                    'silhouette': silhouette_score(X, labels),
                    'Davies-Bouldin': davies_bouldin_score(X, labels),
                    'Calinski-Harabasz': calinski_harabasz_score(X, labels),
                    'partition_coefficient': (membership**2).mean(),
                    'partition_entropy': -(membership * np.log(membership + 1e-10)).mean()
                }
            else:
                metrics = {
                    'silhouette': None,
                    'Davies-Bouldin': None,
                    'Calinski-Harabasz': None,
                    'partition_coefficient': (membership**2).mean(),
                    'partition_entropy': -(membership * np.log(membership + 1e-10)).mean()
                }
            
            results = {
                'method': f'FCM_{version_name}',
                'parameters': {
                    'max_iter': params.max_iter,
                    'm': params.m,
                    'init_method': params.init_method
                },
                'centroids': centroids,
                'metrics': metrics,
                'membership_matrix': membership,
                'n_clusters': unique_clusters
            }
            
            all_results.append(results)
            all_labels.append(labels)
            all_memberships.append(membership)
            
        except Exception as e:
            print(f"Error running FCM {version_name}: {str(e)}")
            continue
    
    return all_results, all_labels, all_memberships