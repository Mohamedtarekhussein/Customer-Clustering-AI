# Customer Clustering AI

A comprehensive platform for customer segmentation and analysis, implementing advanced clustering algorithms with interactive visualizations and real-time comparisons.

## 🚀 Features

- **Multiple Clustering Algorithms**:
  - K-means Variants (Standard, Bisecting, K-means++)
  - DBSCAN
  - Genetic Algorithm (GA)
  - Ant Colony Optimization (ACO)
  - Particle Swarm Optimization (PSO)
  - K-Medoids
  - Fuzzy C-Means
- **Advanced Analysis Tools**:
  - Automatic optimal cluster detection via Elbow Method
  - Comprehensive metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
  - Interactive visualizations for cluster exploration
  - Real-time algorithm performance comparison
  - Exportable results in CSV format
- **Data Processing**:
  - Automatic encoding of categorical variables
  - Feature scaling with StandardScaler
  - PCA for high-dimensional data visualization
- **User-Friendly Interface**:
  - Streamlit-based dashboard
  - Sidebar for feature and algorithm selection
  - Responsive design with custom CSS styling

## 📋 Requirements

```txt
Python>=3.8
streamlit>=1.10.0
pandas>=1.3.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.60.0
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Customer-Clustering-AI.git
   cd Customer-Clustering-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the following directory structure:
   ```
   ├── app.py
   ├── algorithms/
   │   ├── Kmeans.py
   │   ├── KMeansPlusPlus.py
   │   ├── DBSCAN.py
   │   ├── GeneticAlgorithm.py
   │   ├── AntColonyOptimization.py
   │   ├── ParticleSwarmOptimization.py
   │   ├── Kmedoids.py
   │   ├── FuzzyCMeans.py
   │   ├── ClusteringUtils.py
   ├── images/
   └── README.md
   ```

## 💻 Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Access the dashboard at `http://localhost:8501`.
3. Upload a CSV file with numerical or categorical data.
4. Select features, algorithms, and parameters via the sidebar.
5. Explore results, including visualizations, metrics, and comparisons.

## 📊 Algorithm Versions

### K-means Variants
- **Standard**: Classic K-Means with k-means++ initialization
- **Hierarchical**: Agglomerative clustering with Ward linkage
- **Bisecting**: Iterative cluster splitting
- **K-means++**: Enhanced initialization with multiple configurations (Fast, Balanced, Thorough, Precise)

### DBSCAN
- **Conservative**: Larger eps, higher min_samples
- **Aggressive**: Smaller eps, lower min_samples
- **Balanced**: Moderate parameters
- **Dense**: High-density clustering

### Genetic Algorithm
- **Small Population Fast**: Quick runs with smaller population
- **Balanced**: Standard GA settings
- **Large Population Long**: Extensive search with larger population
- **Custom**: Tweaked crossover and mutation rates

### Ant Colony Optimization
- **Quick**: Minimal ants and iterations
- **Balanced**: Standard ACO parameters
- **Thorough**: More ants and iterations
- **Specialized**: Optimized pheromone and heuristic settings

### Particle Swarm Optimization
- **Quick**: Fast exploration with fewer particles
- **Balanced**: Standard PSO settings
- **Thorough**: Deep search with more particles
- **Exploratory**: High inertia for broader exploration

### K-Medoids
- **Fast**: Quick convergence with fewer iterations
- **Balanced**: Standard settings with k-means++ initialization
- **Thorough**: Extensive iterations with Manhattan distance
- **Robust**: High tolerance with Manhattan distance

### Fuzzy C-Means
- **Quick**: Fast convergence with higher error tolerance
- **Balanced**: Standard fuzziness and iterations
- **Fine**: Lower fuzziness for precise clustering
- **Fuzzy**: Higher fuzziness for soft clustering

## 📈 Metrics

| Metric                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| Silhouette Score           | Measures intra-cluster cohesion vs. inter-cluster separation (higher better) |
| Davies-Bouldin Index       | Ratio of within-cluster to between-cluster distances (lower better)          |
| Calinski-Harabasz Score    | Ratio of between-cluster to within-cluster dispersion (higher better)        |
| Partition Coefficient      | Fuzzy C-Means specific, measures membership strength (higher better)         |
| Partition Entropy          | Fuzzy C-Means specific, measures membership uncertainty (lower better)       |
| Average Membership Strength| Fuzzy C-Means specific, indicates average cluster assignment certainty       |

## 🔍 Data Preprocessing

- **Categorical Variables**: Automatically encoded using `pd.factorize`.
- **Scaling**: Features scaled with `StandardScaler`.
- **Visualization**: PCA applied for 2D visualization of high-dimensional data.
- **Handling**: Missing values and invalid data trigger user warnings.

## 📊 Visualization Features

- **Cluster Plots**: 2D scatter plots (PCA for high-dimensional data) with centroids/medoids.
- **Elbow Method**: Plots for optimal cluster number detection.
- **Metric Comparison**: Bar plots comparing algorithm performance.
- **Cluster Characteristics**: Box plots of feature distributions per cluster.
- **Cluster Distribution**: Bar plots showing data point distribution across clusters.
- **Fitness History**: Plots for metaheuristic algorithms (GA, ACO, PSO).

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit changes:
   ```bash
   git commit -m "Add YourFeature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a Pull Request.

## 📝 License

Distributed under the MIT License. See `LICENSE` for details.

## 👥 Authors

- Your Name - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- **scikit-learn**: For robust machine learning implementations
- **Streamlit**: For the intuitive web framework
- **Community**: Contributors and testers for valuable feedback

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com  
Project Link: [https://github.com/yourusername/Customer-Clustering-AI](https://github.com/yourusername/Customer-Clustering-AI)

---
⌨️ with ❤️ by [Your Name](https://github.com/yourusername)