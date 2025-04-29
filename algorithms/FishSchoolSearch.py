import numpy as np
from tqdm import tqdm
from algorithms.ClusteringUtils import *
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def calculate_fitness_FSS(X, centroids, labels=None):
    """Calculate clustering fitness (lower is better) - we use within-cluster sum of squares"""
    if labels is None:
        labels = assign_clusters(X, centroids)

    distances = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:  # Avoid empty clusters
            distances += np.sum((cluster_points - centroids[i])**2)

    return distances

def fss_clustering(X, n_clusters, n_fish=80, n_iterations=150, step_ind_init=0.7, step_vol_init=0.7):
    """Fish School Search algorithm for clustering"""
    # Data dimensions
    n_samples, n_features = X.shape
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    # Initialize fish school (each fish is a set of centroids)
    school = []
    for _ in range(n_fish):
        centroids = np.array([np.random.uniform(X_min, X_max) for _ in range(n_clusters)])
        school.append({'centroids': centroids, 'weight': 1.0, 'fitness': float('inf'), 'delta_centroids': None})

    best_fitness = float('inf')
    best_centroids = None

    # FSS loop
    for iteration in tqdm(range(n_iterations), desc="FSS Progress"):
        # Evaluate fitness for all fish
        for fish in school:
            fish['fitness'] = calculate_fitness_FSS(X, fish['centroids'])

        # Keep track of best solution
        min_idx = np.argmin([fish['fitness'] for fish in school])
        if school[min_idx]['fitness'] < best_fitness:
            best_fitness = school[min_idx]['fitness']
            best_centroids = school[min_idx]['centroids'].copy()

        # Individual movement (Enhanced)
        for fish in school:
            perturbation = np.random.normal(0, step_ind_init, fish['centroids'].shape)
            new_centroids = fish['centroids'] + perturbation
            new_centroids = np.clip(new_centroids, X_min, X_max)
            new_fitness = calculate_fitness_FSS(X, new_centroids)

            # Accept move if fitness improves
            if new_fitness < fish['fitness']:
                fish['delta_centroids'] = new_centroids - fish['centroids']
                fish['centroids'] = new_centroids
                fish['fitness'] = new_fitness
            else:
                fish['delta_centroids'] = np.zeros_like(fish['centroids'])

        # Feeding: Update weights based on fitness improvement
        max_delta_fitness = 0
        for fish in school:
            current_fitness = calculate_fitness_FSS(X, fish['centroids'])
            fish['delta_fitness'] = max(0, fish['fitness'] - current_fitness)  # only positive improvements
            max_delta_fitness = max(max_delta_fitness, fish['delta_fitness'])

        for fish in school:
            if max_delta_fitness > 0:
                fish['weight'] += fish['delta_fitness'] / max_delta_fitness
                fish['weight'] = np.clip(fish['weight'], 1.0, 1000.0)

        # Collective-instinctive movement
        total_weight = sum(fish['weight'] for fish in school)
        weighted_sum_delta = np.zeros((n_clusters, n_features))
        for fish in school:
            if fish['delta_centroids'] is not None:
                weighted_sum_delta += fish['delta_centroids'] * fish['weight']

        if total_weight > 0:
            avg_displacement = weighted_sum_delta / total_weight
            for fish in school:
                fish['centroids'] += avg_displacement
                fish['centroids'] = np.clip(fish['centroids'], X_min, X_max)
                fish['fitness'] = calculate_fitness_FSS(X, fish['centroids'])

        # Collective-volitive movement (enhanced with stronger bias)
        barycenter = np.zeros((n_clusters, n_features))
        for fish in school:
            barycenter += fish['centroids'] * fish['weight']
        barycenter /= total_weight if total_weight > 0 else 1.0

        prev_total_weight = total_weight
        current_total_weight = sum(fish['weight'] for fish in school)

        for fish in school:
            for c in range(n_clusters):
                norm = np.linalg.norm(barycenter[c] - fish['centroids'][c]) + 1e-8  # slight stabilization
                if current_total_weight > prev_total_weight:
                    # Move toward barycenter
                    direction = (barycenter[c] - fish['centroids'][c]) / norm
                else:
                    # Move away from barycenter
                    direction = (fish['centroids'][c] - barycenter[c]) / norm
                fish['centroids'][c] += 1.5 * step_vol_init * np.random.uniform(0, 1) * direction
            fish['centroids'] = np.clip(fish['centroids'], X_min, X_max)
            fish['fitness'] = calculate_fitness_FSS(X, fish['centroids'])

        # Update step sizes (slower decay)
        step_ind_init *= (1 - 0.7 * iteration / n_iterations)
        step_vol_init *= (1 - 0.7 * iteration / n_iterations)

    # Final assignment
    final_labels = assign_clusters(X, best_centroids)

    return best_centroids, final_labels

def run_fss_clustering(X, n_clusters, scaler=None, feature_names=None):
    """
    Run FSS clustering and return results
    
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
    print("\nRunning Fish School Search (FSS) for clustering...")
    fss_centroids, fss_labels = fss_clustering(X, n_clusters)
    
    # Ensure labels are 1D array
    fss_labels = np.ravel(fss_labels)
    
    # Evaluate clustering performance
    results = {
        'method': 'FSS',
        'centroids': fss_centroids,
        'metrics': {
            'silhouette': silhouette_score(X, fss_labels),
            'db_index': davies_bouldin_score(X, fss_labels),
            'ch_score': calinski_harabasz_score(X, fss_labels)
        }
    }
    
    return results, fss_labels
