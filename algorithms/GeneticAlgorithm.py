#changed
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

@dataclass
class GAParameters:
    population_size: int = 50
    n_generations: int = 100
    mutation_rate: float = 0.1
    tournament_size: int = 2
    elite_size: int = 2
    crossover_rate: float = 0.8
    verbose: bool = True

class GeneticAlgorithmClustering:
    def __init__(self, params: GAParameters = None):
        self.params = params or GAParameters()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
    def calculate_fitness(self, X: np.ndarray, centroids: np.ndarray) -> float:
        """Calculate fitness using sum of squared distances"""
        distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
        return np.sum(np.min(distances, axis=1))
    
    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid"""
        distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def initialize_population(self, X: np.ndarray, n_clusters: int) -> List[np.ndarray]:
        """Initialize population with random centroids"""
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        population = []
        
        for _ in range(self.params.population_size):
            centroids = np.array([
                np.random.uniform(X_min, X_max) 
                for _ in range(n_clusters)
            ])
            population.append(centroids)
        return population
    
    def tournament_selection(self, population: List[np.ndarray], 
                        fitness_values: List[float]) -> np.ndarray:
        """Select individual using tournament selection"""
        tournament_idx = np.random.choice(
            len(population), 
            self.params.tournament_size, 
            replace=False
        )
        tournament_fitness = [fitness_values[idx] for idx in tournament_idx]
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform single-point crossover"""
        if np.random.random() < self.params.crossover_rate:
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual: np.ndarray, X_min: np.ndarray, 
                X_max: np.ndarray) -> np.ndarray:
        """Perform mutation on an individual"""
        if np.random.random() < self.params.mutation_rate:
            centroid_idx = np.random.randint(len(individual))
            feature_idx = np.random.randint(individual.shape[1])
            individual[centroid_idx, feature_idx] = np.random.uniform(
                X_min[feature_idx], 
                X_max[feature_idx]
            )
        return individual
    
    def fit(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        """Main clustering method"""
        # Initialize
        population = self.initialize_population(X, n_clusters)
        X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
        
        # Evolution loop
        iterator = range(self.params.n_generations)
        if self.params.verbose:
            iterator = tqdm(iterator, desc="GA Progress")
            
        for _ in iterator:
            # Calculate fitness for all individuals
            fitness_values = [self.calculate_fitness(X, ind) for ind in population]
            
            # Store best solution
            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness_values[min_fitness_idx]
                self.best_solution = population[min_fitness_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            # Create new population
            new_population = []
            
            # Elitism
            elite_indices = np.argsort(fitness_values)[:self.params.elite_size]
            new_population.extend([population[idx].copy() for idx in elite_indices])
            
            # Fill rest of population
            while len(new_population) < self.params.population_size:
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate children
                child1 = self.mutate(child1, X_min, X_max)
                child2 = self.mutate(child2, X_min, X_max)
                
                new_population.append(child1)
                if len(new_population) < self.params.population_size:
                    new_population.append(child2)
            
            population = new_population
        
        # Get final labels
        final_labels = self.assign_clusters(X, self.best_solution)
        return self.best_solution, final_labels

def run_ga_clustering(X: np.ndarray, n_clusters: int) -> Tuple[List[Dict], List[np.ndarray]]:
    """Run GA clustering with 4 different parameter settings and return their results"""
    
    # Define 4 different parameter settings
    parameter_versions = {
        'version_1': GAParameters(
            population_size=30,
            n_generations=50,
            mutation_rate=0.15,
            tournament_size=2,
            elite_size=1,
            crossover_rate=0.9,
            verbose=True
        ),
        'version_2': GAParameters(
            population_size=50,
            n_generations=100,
            mutation_rate=0.1,
            tournament_size=3,
            elite_size=2,
            crossover_rate=0.8,
            verbose=True
        ),
        'version_3': GAParameters(
            population_size=100,
            n_generations=200,
            mutation_rate=0.05,
            tournament_size=4,
            elite_size=3,
            crossover_rate=0.7,
            verbose=True
        ),
        'version_4': GAParameters(
            population_size=80,
            n_generations=150,
            mutation_rate=0.08,
            tournament_size=3,
            elite_size=2,
            crossover_rate=0.75,
            verbose=True
        )
    }
    
    all_results = []
    all_labels = []
    
    # Run GA with each parameter version
    for version_name, params in parameter_versions.items():
        # Initialize and run GA
        ga = GeneticAlgorithmClustering(params)
        centroids, labels = ga.fit(X, n_clusters)
        
        # Calculate metrics
        results = {
            'method': f'GA_{version_name}',
            'centroids': centroids,
            'fitness_history': ga.fitness_history,
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

