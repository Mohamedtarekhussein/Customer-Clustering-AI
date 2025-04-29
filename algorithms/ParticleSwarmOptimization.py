import numpy as np
from tqdm import tqdm
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

def pso_clustering(X, n_clusters, n_particles=30, n_iterations=100, w=0.72, c1=1.49, c2=1.49, verbose=True):
    """
    Perform clustering using Particle Swarm Optimization
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data
    n_clusters : int
        Number of clusters to find
    n_particles : int, optional (default=30)
        Number of particles in the swarm
    n_iterations : int, optional (default=100)
        Number of iterations
    w : float, optional (default=0.72)
        Inertia weight
    c1 : float, optional (default=1.49)
        Cognitive parameter
    c2 : float, optional (default=1.49)
        Social parameter
        
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
    
    # Initialize particles (each particle is a set of centroids)
    particles = []
    velocities = []
    personal_best_positions = []
    personal_best_scores = []
    
    for _ in range(n_particles):
        # Initialize position (centroids)
        position = np.array([np.random.uniform(X_min, X_max) for _ in range(n_clusters)])
        particles.append(position)
        
        # Initialize velocity
        velocity = np.random.uniform(-0.1, 0.1, size=(n_clusters, n_features))
        velocities.append(velocity)
        
        # Initialize personal best
        personal_best_positions.append(position.copy())
        personal_best_scores.append(calculate_fitness(X, position))
    
    # Initialize global best
    global_best_idx = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_score = personal_best_scores[global_best_idx]
    
    # PSO loop
    iterator = range(n_iterations)
    if verbose:
        iterator = tqdm(iterator, desc="PSO Progress")
    
    for _ in iterator:
        for i in range(n_particles):
            # Update velocity
            r1, r2 = np.random.random(size=(n_clusters, n_features)), np.random.random(size=(n_clusters, n_features))
            cognitive_component = c1 * r1 * (personal_best_positions[i] - particles[i])
            social_component = c2 * r2 * (global_best_position - particles[i])
            velocities[i] = w * velocities[i] + cognitive_component + social_component
            
            # Update position
            particles[i] += velocities[i]
            
            # Ensure within bounds
            particles[i] = np.clip(particles[i], X_min, X_max)
            
            # Evaluate fitness
            fitness = calculate_fitness(X, particles[i])
            
            # Update personal best
            if fitness < personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = particles[i].copy()
                
                # Update global best
                if fitness < global_best_score:
                    global_best_score = fitness
                    global_best_position = particles[i].copy()
    
    # Final assignment
    final_labels = assign_clusters(X, global_best_position)
    
    return global_best_position, final_labels

def run_pso_clustering(X, n_clusters, scaler=None, feature_names=None):
    """
    Run PSO clustering and return results
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data
    n_clusters : int
        Number of clusters to find
    scaler : object, optional
        Scaler used to transform the data
    feature_names : list, optional
        Names of the features
        
    Returns:
    --------
    results : dict
        Dictionary containing clustering results and metrics
    labels : array, shape (n_samples,)
        Cluster labels for each point
    """
    print("\nRunning Particle Swarm Optimization (PSO) for clustering...")
    pso_centroids, pso_labels = pso_clustering(X, n_clusters)
    
    # Ensure labels are 1D array
    pso_labels = np.ravel(pso_labels)
    
    # Evaluate clustering performance
    results = {
        'method': 'PSO',
        'centroids': pso_centroids,
        'metrics': {
            'silhouette': silhouette_score(X, pso_labels),
            'db_index': davies_bouldin_score(X, pso_labels),
            'ch_score': calinski_harabasz_score(X, pso_labels)
        }
    }
    
    # Visualize clusters
    visualize_clusters(X, pso_labels, pso_centroids, "Particle Swarm Optimization", scaler, feature_names)
    
    return results, pso_labels

