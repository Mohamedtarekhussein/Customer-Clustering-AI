import numpy as np
from tqdm import tqdm
from algorithms.ClusteringUtils import *
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



def gsa_clustering(X, n_clusters, n_agents=30, n_iterations=100, G0=100, alpha=20):
    """Gravitational Search Algorithm for clustering"""
    # Data dimensions
    n_samples, n_features = X.shape
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    
    # Flatten centroids for GSA (each agent is a flattened set of centroids)
    solution_dim = n_clusters * n_features
    
    # Initialize agents
    agents = []
    velocities = []
    
    for _ in range(n_agents):
        # Create random centroids and flatten
        centroids = np.array([np.random.uniform(X_min, X_max) for _ in range(n_clusters)])
        agent = centroids.flatten()
        agents.append(agent)
        
        # Initialize velocity
        velocity = np.zeros(solution_dim)
        velocities.append(velocity)
    
    agents = np.array(agents)
    velocities = np.array(velocities)
    
    # Evaluate initial fitness
    fitness_values = []
    for agent in agents:
        centroids = agent.reshape(n_clusters, n_features)
        fitness_values.append(calculate_fitness(X, centroids))
    
    best_idx = np.argmin(fitness_values)
    best_solution = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    
    # GSA loop
    for iteration in tqdm(range(n_iterations), desc="GSA Progress"):
        # Calculate gravitational constant
        G = G0 * np.exp(-alpha * iteration / n_iterations)
        
        # Calculate mass for each agent
        worst_fitness = max(fitness_values)
        best_fitness_iter = min(fitness_values)
        
        if worst_fitness == best_fitness_iter:
            masses = np.ones(n_agents)
        else:
            masses = (worst_fitness - np.array(fitness_values)) / (worst_fitness - best_fitness_iter)
        
        # Normalize masses
        masses = masses / np.sum(masses)
        
        # Calculate forces and accelerations
        accelerations = np.zeros_like(agents)
        
        for i in range(n_agents):
            force = np.zeros(solution_dim)
            
            for j in range(n_agents):
                if i != j:
                    # Calculate Euclidean distance
                    R = np.linalg.norm(agents[i] - agents[j])
                    
                    # Calculate gravitational force
                    epsilon = 1e-10  # Small constant to avoid division by zero
                    force_magnitude = G * (masses[i] * masses[j]) / (R + epsilon)
                    
                    # Direction of force
                    force_direction = agents[j] - agents[i]
                    
                    # Random component
                    rand = np.random.random()
                    
                    # Update force
                    force += rand * force_magnitude * force_direction
            
            # Calculate acceleration
            accelerations[i] = force / (masses[i] + 1e-10)
        
        # Update velocities and positions
        velocities = np.random.random() * velocities + accelerations
        agents += velocities
        
        # Ensure bounds
        for i in range(n_agents):
            agents[i] = np.clip(agents[i], np.tile(X_min, n_clusters), np.tile(X_max, n_clusters))
        
        # Evaluate fitness
        for i in range(n_agents):
            centroids = agents[i].reshape(n_clusters, n_features)
            fitness_values[i] = calculate_fitness(X, centroids)
        
        # Update best solution
        current_best_idx = np.argmin(fitness_values)
        if fitness_values[current_best_idx] < best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_solution = agents[current_best_idx].copy()
    
    # Reshape best solution to centroids
    best_centroids = best_solution.reshape(n_clusters, n_features)
    
    # Final assignment
    final_labels = assign_clusters(X, best_centroids)
    
    return best_centroids, final_labels
