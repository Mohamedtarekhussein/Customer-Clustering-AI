import numpy as np
from tqdm import tqdm
from algorithms.ClusteringUtils import *
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def de_clustering(X, n_clusters, pop_size=30, n_generations=100, F=0.8, CR=0.9):
    """Differential Evolution for clustering"""
    # Data dimensions
    n_samples, n_features = X.shape
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    
    # Flatten centroids for DE (each individual is a flattened set of centroids)
    solution_dim = n_clusters * n_features
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        # Create random centroids and flatten
        centroids = np.array([np.random.uniform(X_min, X_max) for _ in range(n_clusters)])
        individual = centroids.flatten()
        population.append(individual)
    
    population = np.array(population)
    
    # Evaluate initial population
    fitness_values = []
    for individual in population:
        centroids = individual.reshape(n_clusters, n_features)
        fitness_values.append(calculate_fitness(X, centroids))
    
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    
    # DE loop
    for generation in tqdm(range(n_generations), desc="DE Progress"):
        for i in range(pop_size):
            # Select three random individuals, different from i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Create mutant vector
            mutant = population[a] + F * (population[b] - population[c])
            
            # Ensure bounds
            mutant = np.clip(mutant, np.tile(X_min, n_clusters), np.tile(X_max, n_clusters))
            
            # Crossover
            trial = np.zeros_like(population[i])
            for j in range(solution_dim):
                if np.random.random() < CR or j == np.random.randint(solution_dim):
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i][j]
            
            # Reshape for fitness calculation
            trial_centroids = trial.reshape(n_clusters, n_features)
            trial_fitness = calculate_fitness(X, trial_centroids)
            
            # Selection
            if trial_fitness < fitness_values[i]:
                population[i] = trial
                fitness_values[i] = trial_fitness
                
                # Update best solution
                if trial_fitness < best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness
    
    # Reshape best solution to centroids
    best_centroids = best_solution.reshape(n_clusters, n_features)
    
    # Final assignment
    final_labels = assign_clusters(X, best_centroids)
    
    return best_centroids, final_labels


