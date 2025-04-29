from tqdm import tqdm
import numpy as np
from algorithms.ClusteringUtils import *
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def calculate_fitness(X, centroids):
    """Calculate the sum of squared distances from each point to its nearest centroid"""
    distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
    return np.sum(np.min(distances, axis=1))

def assign_clusters(X, centroids):
    """Assign each data point to the nearest centroid"""
    distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
    return np.argmin(distances, axis=1)

def abc_clustering(X, n_clusters, colony_size=30, n_iterations=100, limit=20):
    """
    Perform clustering using Artificial Bee Colony algorithm
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data
    n_clusters : int
        Number of clusters to find
    colony_size : int, optional (default=30)
        Size of the bee colony
    n_iterations : int, optional (default=100)
        Number of iterations
    limit : int, optional (default=20)
        Maximum number of trials before abandoning a food source
        
    Returns:
    --------
    centroids : array, shape (n_clusters, n_features)
        The final centroids
    labels : array, shape (n_samples,)
        The cluster labels for each point
    """
    # Data dimensions
    n_samples, n_features = X.shape
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    
    # Number of employed bees (food sources) = number of onlooker bees
    n_employed = colony_size // 2
    
    # Initialize food sources (each is a set of centroids)
    food_sources = []
    fitness_values = []
    trial_counters = np.zeros(n_employed)
    
    for _ in range(n_employed):
        # Initialize random centroids
        centroids = np.array([np.random.uniform(X_min, X_max) for _ in range(n_clusters)])
        food_sources.append(centroids)
        
        # Calculate fitness (we convert to maximization)
        fitness = 1 / (1 + calculate_fitness(X, centroids))
        fitness_values.append(fitness)
    
    best_source_idx = np.argmax(fitness_values)
    best_centroids = food_sources[best_source_idx].copy()
    best_fitness = 1 / (fitness_values[best_source_idx] + 1e-10) - 1  # Convert back to minimization
    
    # ABC loop
    for iteration in tqdm(range(n_iterations), desc="ABC Progress"):
        # Employed Bee Phase
        for i in range(n_employed):
            # Generate a neighbor solution
            neighbor = food_sources[i].copy()
            
            # Modify one dimension of one centroid
            centroid_idx = np.random.randint(n_clusters)
            feature_idx = np.random.randint(n_features)
            
            # Choose another food source to interact with
            j = i
            while j == i:
                j = np.random.randint(n_employed)
            
            # Create a new position using the ABC formula
            phi = np.random.uniform(-1, 1)
            neighbor[centroid_idx, feature_idx] = food_sources[i][centroid_idx, feature_idx] + \
                                               phi * (food_sources[i][centroid_idx, feature_idx] - food_sources[j][centroid_idx, feature_idx])
            
            # Ensure within bounds
            neighbor[centroid_idx, feature_idx] = max(X_min[feature_idx], min(X_max[feature_idx], neighbor[centroid_idx, feature_idx]))
            
            # Calculate fitness
            neighbor_fitness = 1 / (1 + calculate_fitness(X, neighbor))
            
            # Greedy selection
            if neighbor_fitness > fitness_values[i]:
                food_sources[i] = neighbor
                fitness_values[i] = neighbor_fitness
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
        
        # Calculate selection probabilities for onlooker bees
        total_fitness = sum(fitness_values)
        probabilities = [fit / total_fitness for fit in fitness_values]
        
        # Onlooker Bee Phase
        for _ in range(n_employed):
            # Select a food source
            i = np.random.choice(n_employed, p=probabilities)
            
            # Generate a neighbor solution
            neighbor = food_sources[i].copy()
            
            # Modify one dimension of one centroid
            centroid_idx = np.random.randint(n_clusters)
            feature_idx = np.random.randint(n_features)
            j = i
            while j == i:
                j = np.random.randint(n_employed)
            
            # Create a new position using the ABC formula
            phi = np.random.uniform(-1, 1)
            neighbor[centroid_idx, feature_idx] = food_sources[i][centroid_idx, feature_idx] + \
                                               phi * (food_sources[i][centroid_idx, feature_idx] - food_sources[j][centroid_idx, feature_idx])
            
            # Ensure within bounds
            neighbor[centroid_idx, feature_idx] = max(X_min[feature_idx], min(X_max[feature_idx], neighbor[centroid_idx, feature_idx]))
            
            # Calculate fitness
            neighbor_fitness = 1 / (1 + calculate_fitness(X, neighbor))
            
            # Greedy selection
            if neighbor_fitness > fitness_values[i]:
                food_sources[i] = neighbor
                fitness_values[i] = neighbor_fitness
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
        
        # Scout Bee Phase
        for i in range(n_employed):
            if trial_counters[i] >= limit:
                # Abandon food source and create a new one
                new_centroids = np.array([np.random.uniform(X_min, X_max) for _ in range(n_clusters)])
                food_sources[i] = new_centroids
                
                # Calculate fitness
                new_fitness = 1 / (1 + calculate_fitness(X, new_centroids))
                fitness_values[i] = new_fitness
                trial_counters[i] = 0
        
        # Update best solution
        current_best_idx = np.argmax(fitness_values)
        current_best_fitness = 1 / (fitness_values[current_best_idx] + 1e-10) - 1  # Convert back to minimization
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_centroids = food_sources[current_best_idx].copy()
    
    # Final assignment
    final_labels = assign_clusters(X, best_centroids)
    
    return best_centroids, final_labels
