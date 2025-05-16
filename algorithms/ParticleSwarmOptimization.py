import numpy as np
from tqdm import tqdm
from algorithms.ClusteringUtils import *
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class PSOParameters:
    n_clusters: int
    n_particles: int = 30
    n_iterations: int = 100
    w: float = 0.72  # inertia weight
    c1: float = 1.49  # cognitive parameter
    c2: float = 1.49  # social parameter
    verbose: bool = True

def calculate_fitness(X: np.ndarray, centroids: np.ndarray) -> float:
    """Calculate the sum of squared distances from each point to its nearest centroid"""
    distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
    return np.sum(np.min(distances, axis=1))

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each data point to the nearest centroid"""
    distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
    return np.argmin(distances, axis=1)

def pso_clustering_single(X: np.ndarray, params: PSOParameters) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Single run of PSO clustering with specific parameters"""
    n_samples, n_features = X.shape
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    
    # Initialize particles
    particles = []
    velocities = []
    personal_best_positions = []
    personal_best_scores = []
    fitness_history = []
    
    for _ in range(params.n_particles):
        position = np.array([np.random.uniform(X_min, X_max) 
                        for _ in range(params.n_clusters)])
        particles.append(position)
        velocities.append(np.random.uniform(-0.1, 0.1, 
                                        size=(params.n_clusters, n_features)))
        personal_best_positions.append(position.copy())
        personal_best_scores.append(calculate_fitness(X, position))
    
    # Initialize global best
    global_best_idx = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_score = personal_best_scores[global_best_idx]
    
    # PSO loop
    iterator = range(params.n_iterations)
    if params.verbose:
        iterator = tqdm(iterator, desc="PSO Progress")
    
    for _ in iterator:
        for i in range(params.n_particles):
            # Update velocity
            r1 = np.random.random(size=(params.n_clusters, n_features))
            r2 = np.random.random(size=(params.n_clusters, n_features))
            cognitive = params.c1 * r1 * (personal_best_positions[i] - particles[i])
            social = params.c2 * r2 * (global_best_position - particles[i])
            velocities[i] = params.w * velocities[i] + cognitive + social
            
            # Update position
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], X_min, X_max)
            
            # Update personal and global best
            fitness = calculate_fitness(X, particles[i])
            if fitness < personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = particles[i].copy()
                if fitness < global_best_score:
                    global_best_score = fitness
                    global_best_position = particles[i].copy()
        
        fitness_history.append(global_best_score)
    
    final_labels = assign_clusters(X, global_best_position)
    return global_best_position, final_labels, fitness_history

def run_pso_clustering(X: np.ndarray, n_clusters: int) -> Tuple[List[Dict], List[np.ndarray]]:
    """Run PSO clustering with 4 different parameter settings"""
    
    parameter_versions = {
        'quick': PSOParameters(
            n_clusters=n_clusters,
            n_particles=20,
            n_iterations=50,
            w=0.8,
            c1=1.5,
            c2=1.5,
            verbose=False
        ),
        'balanced': PSOParameters(
            n_clusters=n_clusters,
            n_particles=30,
            n_iterations=100,
            w=0.72,
            c1=1.49,
            c2=1.49,
            verbose=False
        ),
        'thorough': PSOParameters(
            n_clusters=n_clusters,
            n_particles=50,
            n_iterations=200,
            w=0.6,
            c1=2.0,
            c2=2.0,
            verbose=False
        ),
        'exploratory': PSOParameters(
            n_clusters=n_clusters,
            n_particles=40,
            n_iterations=150,
            w=0.9,
            c1=1.2,
            c2=1.8,
            verbose=False
        )
    }
    
    all_results = []
    all_labels = []
    
    for version_name, params in parameter_versions.items():
        try:
            centroids, labels, fitness_history = pso_clustering_single(X, params)
            
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
                'method': f'PSO_{version_name}',
                'parameters': {
                    'n_particles': params.n_particles,
                    'n_iterations': params.n_iterations,
                    'w': params.w,
                    'c1': params.c1,
                    'c2': params.c2
                },
                'centroids': centroids,
                'metrics': metrics,
                'fitness_history': fitness_history,
                'n_clusters': unique_clusters
            }
            
            all_results.append(results)
            all_labels.append(labels)
            
        except Exception as e:
            print(f"Error running PSO {version_name}: {str(e)}")
            continue
    
    return all_results, all_labels