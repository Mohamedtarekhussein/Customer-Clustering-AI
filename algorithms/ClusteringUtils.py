import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def initialize_centroids(n_clusters, n_features, X_min, X_max):
    """Initialize random centroids within the data range"""
    return np.random.uniform(X_min, X_max, size=(n_clusters, n_features))

def assign_clusters(X, centroids):
    """Assign each data point to the closest centroid"""
    if centroids is None or len(centroids) == 0:
        return np.zeros(X.shape[0], dtype=int)
    distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
    return np.argmin(distances, axis=1)

def calculate_fitness(X, centroids, labels=None):
    """Calculate clustering fitness (lower is better) - we use within-cluster sum of squares"""
    if centroids is None or len(centroids) == 0:
        return float('inf')
    if labels is None:
        labels = assign_clusters(X, centroids)
    
    total_distance = 0.0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:  # Avoid empty clusters
            total_distance += np.sum((cluster_points - centroids[i])**2)
    
    return total_distance

def evaluate_clustering(X, labels, method_name):
    """Evaluate clustering using multiple metrics"""
    try:
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
        else:
            sil_score = 0
            db_score = float('inf')
            ch_score = 0
        
        print(f"\n{method_name} Clustering Evaluation:")
        print(f"Silhouette Score: {sil_score:.4f}")
        print(f"Davies-Bouldin Index: {db_score:.4f}")
        print(f"Calinski-Harabasz Index: {ch_score:.4f}")
        
        return {
            'method': method_name,
            'silhouette': sil_score,
            'Davies-Bouldin': db_score,
            'Calinski-Harabasz': ch_score
        }
    except Exception as e:
        print(f"Error evaluating {method_name}: {str(e)}")
        return {
            'method': method_name,
            'silhouette': 0,
            'Davies-Bouldin': float('inf'),
            'Calinski-Harabasz': 0
        }

def visualize_clusters(X, labels, centroids, algorithm_name):
    """
    Visualize clusters in 2D using PCA if needed.
    
    Parameters:
    - X: Scaled data array (n_samples, n_features)
    - labels: Cluster labels for each point (n_samples,)
    - centroids: Cluster centroids (n_clusters, n_features) or None
    - algorithm_name: Name of the clustering algorithm
    
    Returns:
    - fig: Matplotlib figure object
    """
    try:
        # Validate inputs
        if X is None or labels is None or len(X) == 0 or len(labels) != len(X):
            fig = plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "Invalid data or labels", ha='center', va='center')
            plt.axis('off')
            plt.title(algorithm_name)
            return fig
        
        # Ensure labels are valid
        labels = np.array(labels, dtype=int)
        if len(np.unique(labels)) == 0:
            fig = plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No valid clusters", ha='center', va='center')
            plt.axis('off')
            plt.title(algorithm_name)
            return fig
        
        # Create figure
        fig = plt.figure(figsize=(15, 6))
        n_features = X.shape[1]
        
        if n_features > 2:
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            centroids_pca = pca.transform(centroids) if centroids is not None and len(centroids) > 0 else None
            
            # Create scatter plot
            ax = fig.add_subplot(111)
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
            if centroids_pca is not None:
                ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroids')
            ax.set_title(f'{algorithm_name} Clustering Results (PCA)')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
        else:
            # Use original features for 2D data
            ax = fig.add_subplot(111)
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
            if centroids is not None and len(centroids) > 0:
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
            ax.set_title(f'{algorithm_name} Clustering Results')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error visualizing {algorithm_name}: {str(e)}")
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Visualization failed: {str(e)}", ha='center', va='center')
        plt.axis('off')
        plt.title(algorithm_name)
        return fig

def plot_original_features(df, centroids, scaler):
    """
    Plot original features with cluster centroids
    """
    try:
        if centroids is None or len(centroids) == 0:
            print("No valid centroids for plotting")
            return None
        
        # Inverse transform centroids to original scale
        centroids_original = scaler.inverse_transform(centroids)
        
        # Create pairplot
        fig = plt.figure(figsize=(10, 6))
        sns.pairplot(df, diag_kind='kde')
        
        # Add centroids to the plot
        for i, centroid in enumerate(centroids_original):
            plt.scatter(centroid[0], centroid[1], c='red', marker='X', s=200, label=f'Centroid {i+1}')
        
        plt.title('Original Features with Cluster Centroids')
        plt.legend()
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error in plot_original_features: {str(e)}")
        return None

def plot_elbow_method(X, max_k=10):
    """
    Plot elbow method for determining optimal number of clusters
    """
    from sklearn.cluster import KMeans
    try:
        distortions = []
        K = range(1, max_k + 1)
        
        for k in K:
            kmeanModel = KMeans(n_clusters=k, random_state=42)
            kmeanModel.fit(X)
            distortions.append(kmeanModel.inertia_)
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.tight_layout()
        return fig, distortions
    
    except Exception as e:
        print(f"Error in plot_elbow_method: {str(e)}")
        return None, []

def plot_silhouette_scores(X, max_k=10):
    """
    Plot silhouette scores for different numbers of clusters
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    try:
        silhouette_scores = []
        K = range(2, max_k + 1)
        
        for k in K:
            kmeanModel = KMeans(n_clusters=k, random_state=42)
            labels = kmeanModel.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels))
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(K, silhouette_scores, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores for different k values')
        plt.tight_layout()
        return fig, silhouette_scores
    
    except Exception as e:
        print(f"Error in plot_silhouette_scores: {str(e)}")
        return None, []

def compare_results(results):
    """Compare multiple clustering results"""
    if not results:
        print("No results to compare")
        return
    
    try:
        print("\n=== Clustering Methods Comparison ===")
        print("{:<15} {:<15} {:<15} {:<15}".format(
            "Method", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"))
        
        for res in results:
            print("{:<15} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                res['method'],
                res['silhouette'],
                res['Davies-Bouldin'],
                res['calinski_harabasz']))
        
        # Find best method by silhouette score (higher is better)
        best_silhouette = max(results, key=lambda x: x['silhouette'])
        # Find best method by Davies-Bouldin (lower is better)
        best_db = min(results, key=lambda x: x['Davies-Bouldin'])
        # Find best method by Calinski-Harabasz (higher is better)
        best_ch = max(results, key=lambda x: x['calinski_harabasz'])
        
        print("\nBest Methods:")
        print(f"By Silhouette: {best_silhouette['method']} ({best_silhouette['silhouette']:.4f})")
        print(f"By Davies-Bouldin: {best_db['method']} ({best_db['davies_bouldin']:.4f})")
        print(f"By Calinski-Harabasz: {best_ch['method']} ({best_ch['calinski_harabasz']:.4f})")
    
    except Exception as e:
        print(f"Error in compare_results: {str(e)}")

def plot_algorithm_comparison(X, results):
    """
    Plot comparison of different algorithms' clustering results
    """
    try:
        n_algorithms = len(results)
        n_features = X.shape[1]
        
        fig = plt.figure(figsize=(6 * n_algorithms, 6))
        
        for idx, result in enumerate(results):
            algo_name = result['method']
            labels = result.get('labels', np.zeros(X.shape[0], dtype=int))
            centroids = result.get('centroids', None)
            
            if n_features > 2:
                # Apply PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                centroids_pca = pca.transform(centroids) if centroids is not None and len(centroids) > 0 else None
                X_plot = X_pca
                centroids_plot = centroids_pca
                xlabel = 'Principal Component 1'
                ylabel = 'Principal Component 2'
            else:
                X_plot = X
                centroids_plot = centroids
                xlabel = 'Feature 1'
                ylabel = 'Feature 2'
            
            # Create subplot
            ax = fig.add_subplot(1, n_algorithms, idx + 1)
            scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis', alpha=0.6)
            if centroids_plot is not None:
                ax.scatter(centroids_plot[:, 0], centroids_plot[:, 1], c='red', marker='X', s=200, label='Centroids')
            
            # Add metrics to title
            metrics = result.get('metrics', {})
            title = f"{algo_name}\nSilhouette: {metrics.get('silhouette', 0):.3f}\nDB: {metrics.get('Davies-Bouldin', 'N/A'):.3f}\nCH: {metrics.get('Calinski-Harabasz', 0):.3f}"
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error in plot_algorithm_comparison: {str(e)}")
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Comparison plot failed: {str(e)}", ha='center', va='center')
        plt.axis('off')
        return fig

def plot_cluster_characteristics(X, labels, algorithm_name):
    """
    Plot characteristics of each cluster
    """
    try:
        n_clusters = len(np.unique(labels))
        n_features = X.shape[1]
        
        fig = plt.figure(figsize=(15, 5 * n_clusters))
        
        for cluster in range(n_clusters):
            cluster_data = X[labels == cluster]
            
            # Plot boxplot for each feature
            ax = fig.add_subplot(n_clusters, 1, cluster + 1)
            sns.boxplot(data=cluster_data, ax=ax)
            ax.set_title(f'Cluster {cluster + 1} Characteristics - {algorithm_name}')
            ax.set_ylabel('Feature Values')
            ax.set_xlabel('Features')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error in plot_cluster_characteristics: {str(e)}")
        return None

def plot_cluster_distribution(labels, algorithm_name):
    """
    Plot distribution of data points across clusters
    """
    try:
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        fig = plt.figure(figsize=(10, 6))
        bars = plt.bar(unique_labels, counts)
        plt.title(f'Cluster Distribution - {algorithm_name}')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Points')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error in plot_cluster_distribution: {str(e)}")
        return None

def plot_algorithm_metrics(results):
    """
    Plot comparison of clustering metrics across algorithms
    """
    try:
        algorithms = [result['method'] for result in results]
        metrics = {
            'silhouette': 'Silhouette Score',
            'Davies-Bouldin': 'Davies-Bouldin Index',
            'Calinski-Harabasz': 'Calinski-Harabasz Score'
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (metric_key, metric_name) in enumerate(metrics.items()):
            try:
                values = [result['metrics'][metric_key] for result in results]
                
                # Plot bar chart
                bars = axes[idx].bar(algorithms, values)
                axes[idx].set_title(metric_name)
                axes[idx].set_ylabel('Score')
                
                # Rotate x-axis labels for better readability
                axes[idx].set_xticklabels(algorithms, rotation=45, ha='right')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}',
                                ha='center', va='bottom')
            
            except KeyError as e:
                axes[idx].text(0.5, 0.5, f'Error: Missing metric "{metric_key}"',
                            ha='center', va='center')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error in plot_algorithm_metrics: {str(e)}")
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Metrics plot failed: {str(e)}", ha='center', va='center')
        plt.axis('off')
        return fig