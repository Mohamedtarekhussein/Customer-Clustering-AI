from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import numpy as np
from algorithms.ClusteringUtils import *
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



def aco_clustering(X, n_clusters, n_ants=30, n_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5):
    """Ant Colony Optimization for clustering"""
    # Data dimensions
    n_samples, n_features = X.shape
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    
    # Initialize pheromone matrix and heuristic information
    # We'll create a grid in the feature space and use pheromones to guide centroid placement
    grid_size = 10  # Number of grid points per dimension
    
    # Create grid points
    grid_points = []
    for f in range(n_features):
        points = np.linspace(X_min[f], X_max[f], grid_size)
        grid_points.append(points)
    
    # Initialize pheromone matrix (higher value = more desirable location for centroids)
    pheromone = np.ones([grid_size] * n_features)
    
    # Initialize heuristic information (we'll use data density)
    heuristic = np.ones([grid_size] * n_features)
    
    # Calculate data density for heuristic
    # For simplicity, we'll just count nearby points
    for point in X:
        # Find closest grid point
        grid_idx = []
        for f in range(n_features):
            idx = np.argmin(np.abs(grid_points[f] - point[f]))
            grid_idx.append(idx)
        
        # Increment density counter
        heuristic[tuple(grid_idx)] += 1
    
    # Normalize heuristic
    heuristic = heuristic / np.sum(heuristic)
    
    best_centroids = None
    best_fitness = float('inf')
    
    # ACO loop
    for iteration in tqdm(range(n_iterations), desc="ACO Progress"):
        all_centroids = []
        all_fitness = []
        
        # Each ant constructs a solution
        for ant in range(n_ants):
            # Select grid points for centroids based on pheromone and heuristic
            centroids = []
            for _ in range(n_clusters):
                # Calculate selection probability
                prob = (pheromone ** alpha) * (heuristic ** beta)
                prob = prob / np.sum(prob)
                
                # Flatten for random choice
                flat_prob = prob.flatten()
                choice = np.random.choice(len(flat_prob), p=flat_prob)
                
                # Convert back to multi-dimensional index
                idx = np.unravel_index(choice, prob.shape)
                
                # Get actual centroid coordinates
                centroid = np.array([grid_points[f][idx[f]] for f in range(n_features)])
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            
            # Evaluate solution
            fitness = calculate_fitness(X, centroids)
            all_centroids.append(centroids)
            all_fitness.append(fitness)
            
            # Update best solution
            if fitness < best_fitness:
                best_fitness = fitness
                best_centroids = centroids.copy()
        
        # Update pheromones (evaporation)
        pheromone = (1 - evaporation_rate) * pheromone
        
        # Add new pheromones based on solution quality
        for ant in range(n_ants):
            # Amount of pheromone to deposit is inversely proportional to fitness
            deposit = 1.0 / (all_fitness[ant] + 1e-10)
            
            # For each centroid, deposit pheromone at nearest grid point
            for centroid in all_centroids[ant]:
                grid_idx = []
                for f in range(n_features):
                    idx = np.argmin(np.abs(grid_points[f] - centroid[f]))
                    grid_idx.append(idx)
                
                pheromone[tuple(grid_idx)] += deposit
    
    # Final assignment
    final_labels = assign_clusters(X, best_centroids)
    
    return best_centroids, final_labels
