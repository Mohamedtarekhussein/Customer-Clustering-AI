from tqdm import tqdm
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

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

def aco_clustering_single(X: np.ndarray, n_clusters: int, params: ACOParameters) -> Tuple[np.ndarray, np.ndarray, List[float]]:
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
        # Use tuple for proper indexing in the n-dimensional array
        tuple_idx = tuple(grid_idx)
        heuristic[tuple_idx] += 1
    
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
                # Use tuple for proper indexing in the n-dimensional array
                tuple_idx = tuple(grid_idx)
                pheromone[tuple_idx] += deposit
    
    final_labels = assign_clusters(X, best_centroids)
    return best_centroids, final_labels, fitness_history

def calculate_optimal_clusters_aco(X: np.ndarray, max_clusters: int = 10) -> Tuple[plt.Figure, int]:
    
    # Use quick parameters for elbow method to save time
    quick_params = ACOParameters(
        n_ants=10,
        n_iterations=30,
        alpha=1.0,
        beta=1.5,
        evaporation_rate=0.6,
        grid_size=8,
        verbose=False
    )
    
    # Calculate distortion (inertia) for a range of number of clusters
    K = range(1, min(max_clusters+1, X.shape[0]))
    inertias = []
    silhouette_scores = []
    
    for k in range(1, min(max_clusters+1, X.shape[0])):
        if k == 1:
            # For k=1, we can't calculate silhouette score
            # Just calculate a centroid as mean of all points
            centroid = np.mean(X, axis=0).reshape(1, -1)
            labels = np.zeros(X.shape[0], dtype=int)
            inertia = np.sum(np.square(X - centroid))
            inertias.append(inertia)
            silhouette_scores.append(0)  # Not defined for k=1
        else:
            centroids, labels, _ = aco_clustering_single(X, k, quick_params)
            
            # Calculate inertia (sum of squared distances to centroids)
            distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
            inertia = np.sum(np.min(distances, axis=1)**2)
            inertias.append(inertia)
            
            # Calculate silhouette score
            try:
                silhouette_scores.append(silhouette_score(X, labels))
            except:
                silhouette_scores.append(0)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot elbow curve (inertia)
    ax1.plot(K, inertias, 'bo-')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Sum of squared distances')
    ax1.set_title('Elbow Method for Optimal k (ACO)')
    ax1.grid(True)
    
    # Plot silhouette scores
    ax2.plot(K[1:], silhouette_scores[1:], 'ro-')  # Skip k=1 for silhouette
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for Optimal k (ACO)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Determine optimal k using the elbow method
    # Calculate the rate of decrease in inertia
    inertia_diffs = np.diff(inertias)
    inertia_diffs_rate = np.diff(inertia_diffs)
    
    # Find the elbow point (where the rate of decrease significantly changes)
    if len(inertia_diffs_rate) > 0:
        elbow_index = np.argmax(inertia_diffs_rate) + 1
        optimal_k = K[elbow_index]
    else:
        optimal_k = 2  # Default if we can't determine
    
    # Alternatively, use maximum silhouette score
    if any(s > 0 for s in silhouette_scores[1:]):
        sil_optimal_k = K[1:][np.argmax(silhouette_scores[1:])]
    else:
        sil_optimal_k = optimal_k
    
    # Return both the figure and a compromise between the two methods
    final_optimal_k = (optimal_k + sil_optimal_k) // 2
    
    # Add a vertical line indicating the optimal k
    ax1.axvline(x=final_optimal_k, color='r', linestyle='--')
    ax2.axvline(x=final_optimal_k, color='r', linestyle='--')
    
    ax1.text(final_optimal_k, max(inertias)*0.9, f'Optimal k = {final_optimal_k}', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    return fig, final_optimal_k

def run_aco_clustering(X: np.ndarray, n_clusters: int) -> Tuple[List[Dict], List[np.ndarray]]:
    """Run ACO clustering with 4 different parameter settings"""
    
    results_list = []
    labels_list = []
    
    # Version 1: Quick ACO with minimal iterations and ants
    quick_params = ACOParameters(
        n_ants=15,
        n_iterations=30,
        alpha=1.0,
        beta=1.5,
        evaporation_rate=0.6,
        grid_size=8,
        verbose=False
    )
    
    centroids_quick, labels_quick, fitness_history_quick = aco_clustering_single(X, n_clusters, quick_params)
    
    # Calculate metrics
    if len(np.unique(labels_quick)) > 1:  # Check if we have more than one cluster
        sil_score_quick = silhouette_score(X, labels_quick)
        db_score_quick = davies_bouldin_score(X, labels_quick)
        ch_score_quick = calinski_harabasz_score(X, labels_quick)
    else:
        sil_score_quick = 0
        db_score_quick = float('inf')
        ch_score_quick = 0
    
    results_quick = {
        'method': 'ACO_Quick',
        'centroids': centroids_quick,
        'fitness_history': fitness_history_quick,
        'parameters': quick_params,
        'metrics': {
            'silhouette': sil_score_quick,
            'Davies-Bouldin': db_score_quick,
            'Calinski-Harabasz Score': ch_score_quick
        }
    }
    
    results_list.append(results_quick)
    labels_list.append(labels_quick)
    
    # Version 2: Balanced ACO with standard settings
    balanced_params = ACOParameters(
        n_ants=30,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.5,
        grid_size=10,
        verbose=False
    )
    
    centroids_balanced, labels_balanced, fitness_history_balanced = aco_clustering_single(X, n_clusters, balanced_params)
    
    # Calculate metrics
    if len(np.unique(labels_balanced)) > 1:
        sil_score_balanced = silhouette_score(X, labels_balanced)
        db_score_balanced = davies_bouldin_score(X, labels_balanced)
        ch_score_balanced = calinski_harabasz_score(X, labels_balanced)
    else:
        sil_score_balanced = 0
        db_score_balanced = float('inf')
        ch_score_balanced = 0
    
    results_balanced = {
        'method': 'ACO_Balanced',
        'centroids': centroids_balanced,
        'fitness_history': fitness_history_balanced,
        'parameters': balanced_params,
        'metrics': {
            'silhouette': sil_score_balanced,
            'Davies-Bouldin': db_score_balanced,
            'Calinski-Harabasz Score': ch_score_balanced
        }
    }
    
    results_list.append(results_balanced)
    labels_list.append(labels_balanced)
    
    # Version 3: Thorough ACO with more iterations and ants
    thorough_params = ACOParameters(
        n_ants=50,
        n_iterations=200,
        alpha=1.0,
        beta=2.5,
        evaporation_rate=0.3,
        grid_size=12,
        verbose=False
    )
    
    centroids_thorough, labels_thorough, fitness_history_thorough = aco_clustering_single(X, n_clusters, thorough_params)
    
    # Calculate metrics
    if len(np.unique(labels_thorough)) > 1:
        sil_score_thorough = silhouette_score(X, labels_thorough)
        db_score_thorough = davies_bouldin_score(X, labels_thorough)
        ch_score_thorough = calinski_harabasz_score(X, labels_thorough)
    else:
        sil_score_thorough = 0
        db_score_thorough = float('inf')
        ch_score_thorough = 0
    
    results_thorough = {
        'method': 'ACO_Thorough',
        'centroids': centroids_thorough,
        'fitness_history': fitness_history_thorough,
        'parameters': thorough_params,
        'metrics': {
            'silhouette': sil_score_thorough,
            'Davies-Bouldin': db_score_thorough,
            'Calinski-Harabasz Score': ch_score_thorough
        }
    }
    
    results_list.append(results_thorough)
    labels_list.append(labels_thorough)
    
    # Version 4: Specialized ACO with tweaked parameters
    specialized_params = ACOParameters(
        n_ants=40,
        n_iterations=150,
        alpha=1.2,
        beta=2.2,
        evaporation_rate=0.4,
        grid_size=10,
        verbose=False
    )
    
    centroids_specialized, labels_specialized, fitness_history_specialized = aco_clustering_single(X, n_clusters, specialized_params)
    
    # Calculate metrics
    if len(np.unique(labels_specialized)) > 1:
        sil_score_specialized = silhouette_score(X, labels_specialized)
        db_score_specialized = davies_bouldin_score(X, labels_specialized)
        ch_score_specialized = calinski_harabasz_score(X, labels_specialized)
    else:
        sil_score_specialized = 0
        db_score_specialized = float('inf')
        ch_score_specialized = 0
    
    results_specialized = {
        'method': 'ACO_Specialized',
        'centroids': centroids_specialized,
        'fitness_history': fitness_history_specialized,
        'parameters': specialized_params,
        'metrics': {
            'silhouette': sil_score_specialized,
            'Davies-Bouldin': db_score_specialized,
            'Calinski-Harabasz Score': ch_score_specialized
        }
    }
    
    results_list.append(results_specialized)
    labels_list.append(labels_specialized)
    
    return results_list, labels_list