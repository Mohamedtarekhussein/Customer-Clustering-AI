import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Import algorithm implementations (assuming they work)
from algorithms.KMeansPlusPlus import run_kmeans_plusplus
from algorithms.DBSCAN import run_dbscan_clustering
from algorithms.AgglomerativeClustering import run_agglomerative_clustering 
from algorithms.ClusteringUtils import (
    visualize_clusters, plot_algorithm_comparison, plot_cluster_characteristics,
    plot_algorithm_metrics, plot_cluster_distribution
)
from algorithms.ParticleSwarmOptimization import run_pso_clustering
from algorithms.GeneticAlgorithm import run_ga_clustering
from algorithms.ArtificialBeeColony import abc_clustering
from algorithms.AntColonyOptimization import aco_clustering
from algorithms.FishSchoolSearch import run_fss_clustering
from algorithms.GravitationalSearchAlgo import gsa_clustering
from algorithms.DIfferentialEvolution import de_clustering

# Page config
st.set_page_config(page_title="Clustering Algorithms Comparison", layout="wide")

st.title("Clustering Algorithms Comparison Dashboard")
st.sidebar.header("Input Parameters")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Feature selection
        st.sidebar.subheader("Feature Selection")
        features = st.sidebar.multiselect("Select features for clustering", df.columns)

        if features:
            X = df[features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Elbow Method
            if st.sidebar.checkbox("Show Elbow Method Plot"):
                max_k = st.sidebar.slider("Max K for Elbow Method", 2, 15, 10)
                wcss = []
                for i in range(2, max_k+1):
                    kmeans = KMeans(n_clusters=i, random_state=42)
                    kmeans.fit(X_scaled)
                    wcss.append(kmeans.inertia_)

                fig, ax = plt.subplots()
                ax.plot(range(2, max_k+1), wcss, marker='o')
                ax.set_title("Elbow Method")
                ax.set_xlabel("Number of Clusters")
                ax.set_ylabel("WCSS")
                st.pyplot(fig)

            # Algorithm selection
            st.sidebar.subheader("Algorithm Selection")
            algorithms = st.sidebar.multiselect(
                "Select algorithms to compare",
                ["K-means", "K-means++", "Agglomerative", "DBSCAN", "PSO", "GA", "ABC", "ACO", "FSS", "GSA", "DE"]
            )

            if algorithms:
                results = {}
                st.sidebar.subheader("Common Parameters")
                n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

                for algo in algorithms:
                    st.subheader(f"{algo} Clustering Results")
                    try:
                        if algo == "K-means":
                            model = KMeans(n_clusters=n_clusters, random_state=42)
                            labels = model.fit_predict(X_scaled)
                            centroids = model.cluster_centers_

                        elif algo == "K-means++":
                            result_dict, labels = run_kmeans_plusplus(X_scaled, n_clusters, scaler, features)
                            centroids = result_dict['centroids']

                        elif algo == "Agglomerative":
                            
                            result = run_agglomerative_clustering(X_scaled, n_clusters)
                            labels = result['labels']
                            metrics = result['metrics']

                            st.write(f"Silhouette Score: {metrics['silhouette']:.4f}")
                            st.write(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
                            st.write(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f}")

                            visualize_clusters(X_scaled, labels, result['centroids'], method_name='Agglomerative Clustering')

                        

                        elif algo == "DBSCAN":
                            result = run_dbscan_clustering(X_scaled, df)
                            labels = result['labels']
                            metrics = result['metrics']

                            if 'error' not in metrics:
                                st.write(f"Silhouette Score: {metrics['silhouette']:.4f}")
                                st.write(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
                                st.write(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f}")
                            else:
                                st.warning(metrics['error'])

                            visualize_clusters(X_scaled, labels, result['centroids'], method_name='DBSCAN')

# DBSCAN doesn't produce centroids

                        elif algo == "PSO":
                            result_dict, labels = run_pso_clustering(X_scaled, n_clusters, scaler, features)
                            centroids = result_dict['centroids']

                        elif algo == "GA":
                            result_dict, labels = run_ga_clustering(X_scaled, n_clusters, scaler, features)
                            centroids = result_dict['centroids']

                        elif algo == "ABC":
                            centroids, labels = abc_clustering(X_scaled, n_clusters)

                        elif algo == "ACO":
                            centroids, labels = aco_clustering(X_scaled, n_clusters)

                        elif algo == "FSS":
                            result_dict, labels = run_fss_clustering(X_scaled, n_clusters, scaler, features)
                            centroids = result_dict['centroids']

                        elif algo == "GSA":
                            centroids, labels = gsa_clustering(X_scaled, n_clusters)

                        elif algo == "DE":
                            centroids, labels = de_clustering(X_scaled, n_clusters)

                        # Compute metrics
                        metrics = {
                            'silhouette': silhouette_score(X_scaled, labels),
                            'db_index': davies_bouldin_score(X_scaled, labels),
                            'ch_score': calinski_harabasz_score(X_scaled, labels)
                        }

                        # Store results
                        results[algo] = {
                            'labels': labels,
                            'centroids': centroids,
                            'metrics': metrics
                        }

                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(visualize_clusters(X_scaled, labels, centroids, algo))
                        with col2:
                            st.pyplot(plot_cluster_distribution(labels, algo))

                        st.pyplot(plot_cluster_characteristics(X_scaled, labels, algo))

                        st.write("Metrics:")
                        st.write(f"Silhouette Score: {metrics['silhouette']:.4f}")
                        st.write(f"Davies-Bouldin Index: {metrics['db_index']:.4f}")
                        st.write(f"Calinski-Harabasz Score: {metrics['ch_score']:.4f}")

                    except Exception as e:
                        st.error(f"Error running {algo}: {str(e)}")

                # Comparison section
                if results:
                    st.subheader("Algorithm Comparison")
                    st.pyplot(plot_algorithm_comparison(X_scaled, results))
                    st.pyplot(plot_algorithm_metrics(results))

                    comparison_data = [
                        {
                            'Algorithm': algo,
                            'Silhouette Score': res['metrics']['silhouette'],
                            'Davies-Bouldin Index': res['metrics']['db_index'],
                            'Calinski-Harabasz Score': res['metrics']['ch_score']
                        }
                        for algo, res in results.items()
                    ]

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df)

                    st.download_button(
                        label="Download Comparison CSV",
                        data=comparison_df.to_csv(index=False),
                        file_name="clustering_comparison.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

else:
    st.info("Please upload a CSV file to begin.")
