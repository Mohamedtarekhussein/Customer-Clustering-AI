#changed
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import numpy as np
from algorithms.ClusteringUtils import *
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class ACOParameters:
    n_ants: int = 30
    n_iterations: int = 100
    alpha: float = 1.0
    beta: float = 2.0
    evaporation_rate: float = 0.5
    grid_size: int = 10
    verbose: bool = True

def calculate_fitness(X: np.ndarray, centroids: np.ndarray) -> float:
    """Calculate fitness using sum of squared distances"""
    distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
    return np.sum(np.min(distances, axis=1))

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each point to nearest centroid"""
    distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
    return np.argmin(distances, axis=1)

def aco_clustering_single(X: np.ndarray, n_clusters: int, params: ACOParameters) -> Tuple[np.ndarray, np.ndarray]:
    """Single run of ACO clustering with given parameters"""
    n_samples, n_features = X.shape
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    
    # Create grid points
    grid_points = []
    for f in range(n_features):
        points = np.linspace(X_min[f], X_max[f], params.grid_size)
        grid_points.append(points)
    
    pheromone = np.ones([params.grid_size] * n_features)
    heuristic = np.ones([params.grid_size] * n_features)
    
    # Calculate data density for heuristic
    for point in X:
        grid_idx = []
        for f in range(n_features):
            idx = np.argmin(np.abs(grid_points[f] - point[f]))
            grid_idx.append(idx)
        heuristic[tuple(grid_idx)] += 1
    
    heuristic = heuristic / np.sum(heuristic)
    
    best_centroids = None
    best_fitness = float('inf')
    fitness_history = []
    
    iterator = range(params.n_iterations)
    if params.verbose:
        iterator = tqdm(iterator, desc="ACO Progress")
    
    for iteration in iterator:
        all_centroids = []
        all_fitness = []
        
        for ant in range(params.n_ants):
            centroids = []
            for _ in range(n_clusters):
                prob = (pheromone ** params.alpha) * (heuristic ** params.beta)
                prob = prob / np.sum(prob)
                
                flat_prob = prob.flatten()
                choice = np.random.choice(len(flat_prob), p=flat_prob)
                idx = np.unravel_index(choice, prob.shape)
                
                centroid = np.array([grid_points[f][idx[f]] for f in range(n_features)])
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            fitness = calculate_fitness(X, centroids)
            all_centroids.append(centroids)
            all_fitness.append(fitness)
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_centroids = centroids.copy()
        
        fitness_history.append(best_fitness)
        pheromone = (1 - params.evaporation_rate) * pheromone
        
        for ant in range(params.n_ants):
            deposit = 1.0 / (all_fitness[ant] + 1e-10)
            for centroid in all_centroids[ant]:
                grid_idx = []
                for f in range(n_features):
                    idx = np.argmin(np.abs(grid_points[f] - centroid[f]))
                    grid_idx.append(idx)
                pheromone[tuple(grid_idx)] += deposit
    
    final_labels = assign_clusters(X, best_centroids)
    return best_centroids, final_labels, fitness_history

def run_aco_clustering(X: np.ndarray, n_clusters: int) -> Tuple[List[Dict], List[np.ndarray]]:
    """Run ACO clustering with 4 different parameter settings"""
    
    parameter_versions = {
        'version_1': ACOParameters(
            n_ants=20,
            n_iterations=50,
            alpha=1.0,
            beta=1.5,
            evaporation_rate=0.6,
            grid_size=8,
            verbose=True
        ),
        'version_2': ACOParameters(
            n_ants=30,
            n_iterations=100,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.5,
            grid_size=10,
            verbose=True
        ),
        'version_3': ACOParameters(
            n_ants=50,
            n_iterations=200,
            alpha=1.0,
            beta=2.5,
            evaporation_rate=0.3,
            grid_size=12,
            verbose=True
        ),
        'version_4': ACOParameters(
            n_ants=40,
            n_iterations=150,
            alpha=1.2,
            beta=2.2,
            evaporation_rate=0.4,
            grid_size=10,
            verbose=True
        )
    }
    
    all_results = []
    all_labels = []
    
    for version_name, params in parameter_versions.items():
        centroids, labels, fitness_history = aco_clustering_single(X, n_clusters, params)
        
        results = {
            'method': f'ACO_{version_name}',
            'centroids': centroids,
            'fitness_history': fitness_history,
            'parameters': params,
            'metrics': {
                'silhouette': silhouette_score(X, labels),
                'db_index': davies_bouldin_score(X, labels),
                'ch_score': calinski_harabasz_score(X, labels)
            }
        }
        
        all_results.append(results)
        all_labels.append(labels)
    
    return all_results, all_labels