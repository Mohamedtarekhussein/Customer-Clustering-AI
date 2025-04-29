import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def calculate_fitness(X, centroids):
    """Calculate the sum of squared distances from each point to its nearest centroid"""
    distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
    return np.sum(np.min(distances, axis=1))

def assign_clusters(X, centroids):
    """Assign each data point to the nearest centroid"""
    distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
    return np.argmin(distances, axis=1)

def ga_clustering(X, n_clusters, population_size=50, n_generations=100, mutation_rate=0.1, verbose=True):
    """
    Perform clustering using Genetic Algorithm
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data
    n_clusters : int
        Number of clusters to find
    population_size : int, optional (default=50)
        Size of the population
    n_generations : int, optional (default=100)
        Number of generations
    mutation_rate : float, optional (default=0.1)
        Probability of mutation
    verbose : bool, optional (default=True)
        Whether to show progress bar
        
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
    
    # Initialize population
    population = []
    for _ in range(population_size):
        centroids = np.array([np.random.uniform(X_min, X_max) for _ in range(n_clusters)])
        population.append(centroids)
    
    # Evolution loop
    iterator = range(n_generations)
    if verbose:
        iterator = tqdm(iterator, desc="GA Progress")
    
    for _ in iterator:
        # Calculate fitness
        fitness = [calculate_fitness(X, individual) for individual in population]
        
        # Selection (tournament selection)
        new_population = []
        for _ in range(population_size):
            # Select two random individuals
            idx1, idx2 = np.random.choice(population_size, 2, replace=False)
            if fitness[idx1] < fitness[idx2]:
                new_population.append(population[idx1].copy())
            else:
                new_population.append(population[idx2].copy())
        
        # Crossover
        for i in range(0, population_size, 2):
            if i + 1 < population_size:
                parent1 = new_population[i]
                parent2 = new_population[i + 1]
                
                # Single-point crossover
                crossover_point = np.random.randint(1, n_clusters)
                child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
                
                new_population[i] = child1
                new_population[i + 1] = child2
        
        # Mutation
        for i in range(population_size):
            if np.random.random() < mutation_rate:
                # Select a random centroid
                centroid_idx = np.random.randint(n_clusters)
                # Mutate a random feature
                feature_idx = np.random.randint(n_features)
                new_population[i][centroid_idx, feature_idx] = np.random.uniform(
                    X_min[feature_idx], X_max[feature_idx]
                )
        
        population = new_population
    
    # Find best solution
    fitness = [calculate_fitness(X, individual) for individual in population]
    best_idx = np.argmin(fitness)
    best_centroids = population[best_idx]
    
    # Final assignment
    final_labels = assign_clusters(X, best_centroids)
    
    return best_centroids, final_labels

def run_ga_clustering(X, n_clusters, scaler=None, feature_names=None):
    """
    Run GA clustering and return results
    
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
    print("\nRunning Genetic Algorithm (GA) for clustering...")
    ga_centroids, ga_labels = ga_clustering(X, n_clusters)
    
    # Ensure labels are 1D array
    ga_labels = np.ravel(ga_labels)
    
    # Evaluate clustering performance
    results = {
        'method': 'GA',
        'centroids': ga_centroids,
        'metrics': {
            'silhouette': silhouette_score(X, ga_labels),
            'db_index': davies_bouldin_score(X, ga_labels),
            'ch_score': calinski_harabasz_score(X, ga_labels)
        }
    }
    
    return results, ga_labels

