import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Import algorithm implementations
from algorithms.Kmeans import run_kmeans_clustering, calculate_optimal_clusters
from algorithms.DBSCAN import run_dbscan_clustering, DBSCANParameters
from algorithms.GeneticAlgorithm import run_ga_clustering, GAParameters
from algorithms.AntColonyOptimization import run_aco_clustering, aco_clustering_single, ACOParameters, calculate_optimal_clusters_aco
from algorithms.ParticleSwarmOptimization import run_pso_clustering, PSOParameters
from algorithms.Kmedoids import run_kmedoids_clustering, KMedoidsParameters
from algorithms.FuzzyCMeans import run_fcm_clustering, FCMParameters
from algorithms.ClusteringUtils import (
    visualize_clusters, plot_algorithm_comparison, plot_cluster_characteristics,
    plot_algorithm_metrics, plot_cluster_distribution
)

# Configure image saving
STATIC_DIR = "./images"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Page configuration
st.set_page_config(
    page_title="Advanced Clustering Algorithms Comparison",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {padding: 20px;}
    .stButton>button {width: 100%; border-radius: 5px;}
    .stSidebar {background-color: #000000; color: white;}
    .stSpinner {margin: 20px;}
    .metric-box {border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin: 5px 0; background-color: #000000;}
    .fitness-plot {margin-top: 20px;}
    .warning-box {background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0;}
    /* Make sidebar text white */
    .stSidebar .stMarkdown, 
    .stSidebar .stTextInput label, 
    .stSidebar .stNumberInput label,
    .stSidebar .stSelectbox label,
    .stSidebar .stMultiselect label,
    .stSidebar .stSlider label,
    .stSidebar .stCheckbox label {
        color: white !important;
    }
    /* Style the file uploader */
    .stSidebar .stFileUploader label {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Advanced Clustering Algorithms Comparison Dashboard")
st.markdown("Analyze your dataset with multiple clustering algorithms and compare their performance.")

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"], help="Upload a CSV with numerical features.")
    
    # Feature and algorithm selection
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Feature Selection")
        features = st.multiselect(
            "Select Features for Clustering",
            df.columns,
            help="Choose numerical columns for clustering."
        )
        
        st.subheader("Algorithm Selection")
        algorithms = st.multiselect(
            "Select Algorithms to Compare",
            ["K-means Variants", "DBSCAN", "Genetic Algorithm", "Ant Colony", "PSO", "K-Medoids", "Fuzzy C-Means"],
            default=["K-means Variants"],
            help="Select one or more clustering algorithms."
        )
        
        st.subheader("Common Parameters")
        use_elbow_method = st.checkbox(
            "Use Elbow Method to find optimal clusters",
            value=False,
            help="Automatically determine the optimal number of clusters."
        )
        
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=3,
            help="Set the number of clusters (not applicable for DBSCAN)."
        )

# Main content
if uploaded_file:
    try:
        # Dataset preview
        with st.expander("Dataset Preview", expanded=True):
            st.write(df.head())
            st.write(f"Dataset Shape: {df.shape}")

        if not features:
            st.warning("Please select at least one feature for clustering.")
        elif not algorithms:
            st.warning("Please select at least one algorithm.")
        else:
            def preprocess_data(df, features):
                """Preprocess data by encoding categorical variables and scaling numerical ones"""
                try:
                    X = df[features].copy()
                    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

                    # Encode categorical variables
                    for col in categorical_cols:
                        X[col] = pd.factorize(X[col])[0]

                    # Scale all features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    return X_scaled, scaler
                except Exception as e:
                    st.error(f"Error in preprocessing data: {str(e)}")
                    return None, None

            # Run algorithms
            X_scaled, scaler = preprocess_data(df, features)

            if X_scaled is None:
                st.error("Preprocessing failed. Please check your data.")
                st.stop()
                
            # Determine optimal clusters using Elbow method if requested
            if use_elbow_method:
                with st.spinner("Finding optimal number of clusters..."):
                    st.subheader("Elbow Method Analysis")
                    
                    # Check selected algorithms to determine which elbow method to use
                    if len(algorithms) == 1 and algorithms[0] == "Ant Colony":
                        # Use ACO elbow method
                        try:
                            elbow_fig, optimal_k = calculate_optimal_clusters_aco(X_scaled, max_clusters=10)
                            if elbow_fig is None:
                                st.error("Failed to generate elbow plot for ACO.")
                            else:
                                # Save elbow plot
                                elbow_plot_path = os.path.join(STATIC_DIR, "elbow_method_aco.png")
                                plt.savefig(elbow_plot_path)
                                st.image(elbow_plot_path)
                                plt.close(elbow_fig)
                                
                                st.success(f"The optimal number of clusters determined by ACO is: {optimal_k}")
                                n_clusters = optimal_k  # Update the number of clusters
                        except Exception as e:
                            st.error(f"Error in ACO Elbow Method calculation: {str(e)}")
                    else:
                        # Use standard K-means elbow method
                        try:
                            elbow_fig, optimal_k = calculate_optimal_clusters(X_scaled, max_clusters=10)
                            if elbow_fig is None:
                                st.error("Failed to generate elbow plot.")
                            else:
                                # Save elbow plot
                                elbow_plot_path = os.path.join(STATIC_DIR, "elbow_method.png")
                                plt.savefig(elbow_plot_path)
                                st.image(elbow_plot_path)
                                plt.close(elbow_fig)
                                
                                st.success(f"The optimal number of clusters determined is: {optimal_k}")
                                n_clusters = optimal_k  # Update the number of clusters
                        except Exception as e:
                            st.error(f"Error in Elbow Method calculation: {str(e)}")

            all_results = []
            all_labels = []
            with st.spinner("Running clustering algorithms..."):
                for algo in algorithms:
                    st.subheader(f"{algo} Clustering Results")
                    try:
                        # Run algorithm
                        if algo == "K-means Variants":
                            results_list, labels_list = run_kmeans_clustering(X_scaled, n_clusters)
                        elif algo == "DBSCAN":
                            results_list, labels_list = run_dbscan_clustering(X_scaled)
                        elif algo == "Genetic Algorithm":
                            results_list, labels_list = run_ga_clustering(X_scaled, n_clusters)
                        elif algo == "Ant Colony":
                            results_list, labels_list = run_aco_clustering(X_scaled, n_clusters)
                        elif algo == "PSO":
                            results_list, labels_list = run_pso_clustering(X_scaled, n_clusters)
                        elif algo == "K-Medoids":
                            results_list, labels_list = run_kmedoids_clustering(X_scaled, n_clusters)
                        elif algo == "Fuzzy C-Means":
                            results_list, labels_list, memberships_list = run_fcm_clustering(X_scaled, n_clusters)
                            
                        # Display results
                        for idx, result in enumerate(results_list):
                            version_name = result['method']
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.markdown(f"**{version_name}**")
                                metrics = result.get('metrics', {})
                                
                                if isinstance(metrics, dict):
                                    metric_display = []
                                    if metrics.get('silhouette') is not None:
                                        metric_display.append(f"<strong>Silhouette Score:</strong> {metrics['silhouette']:.4f}")
                                    if metrics.get('Davies-Bouldin') is not None:
                                        metric_display.append(f"<strong>Davies-Bouldin:</strong> {metrics['Davies-Bouldin']:.4f}")
                                    if metrics.get('Calinski-Harabasz') is not None:
                                        metric_display.append(f"<strong>Calinski-Harabasz:</strong> {metrics['Calinski-Harabasz']:.4f}")
                                    
                                    if metric_display:
                                        st.markdown(f"""
                                            <div class='metric-box'>
                                                {"<br>".join(metric_display)}
                                            </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown("""
                                            <div class='warning-box'>
                                                Standard metrics unavailable (possibly only one cluster found)
                                            </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.warning(f"Metrics unavailable or invalid for {version_name}.")
                                
                                if 'fitness_history' in result:
                                    st.markdown(f"<div class='metric-box'>Final Fitness: {result['fitness_history'][-1]:.2f}</div>", unsafe_allow_html=True)
                                    st.markdown(f"<div class='metric-box'>Average Fitness: {np.mean(result['fitness_history']):.2f}</div>", unsafe_allow_html=True)

                                
                                if 'membership_matrix' in result:
                                    st.markdown(f"<div class='metric-box'>Average Membership Strength: {np.mean(result['membership_matrix'].max(axis=1)):.3f}</div>", unsafe_allow_html=True)
                                
                                if 'n_clusters' in result:
                                    st.markdown(f"<div class='metric-box'>Actual Clusters Found: {result['n_clusters']}</div>", unsafe_allow_html=True)
                            
                            with col2:
                                try:
                                    plt.figure(figsize=(8, 6))
                                    # Validate labels
                                    if labels_list[idx] is None or not np.any(labels_list[idx]):
                                        st.warning(f"No valid clusters found for {version_name}.")
                                        fig = plt.figure(figsize=(8, 6))
                                        plt.text(0.5, 0.5, "No valid clusters", ha='center', va='center')
                                        plt.axis('off')
                                    else:
                                        if 'centroids' in result and result['centroids'] is not None and len(result['centroids']) > 0:
                                            fig = visualize_clusters(X_scaled, labels_list[idx], result['centroids'], version_name)
                                        elif 'medoid_indices' in result:
                                            medoids = X_scaled[result['medoid_indices']]
                                            fig = visualize_clusters(X_scaled, labels_list[idx], medoids, version_name)
                                        else:
                                            fig = visualize_clusters(X_scaled, labels_list[idx], None, version_name)
                                    
                                    # Save plot to file
                                    plot_path = os.path.join(STATIC_DIR, f"{version_name.replace(' ', '_')}.png")
                                    plt.savefig(plot_path)
                                    st.image(plot_path)
                                    plt.close(fig)
                                except Exception as e:
                                    st.error(f"Error visualizing {version_name}: {str(e)}")
                            
                        all_results.extend(results_list)
                        if algo == "Fuzzy C-Means":
                            all_labels.extend(labels_list)
                        else:
                            all_labels.extend(labels_list)

                    except Exception as e:
                        st.error(f"Error running {algo}: {str(e)}")

            # Comparison section
            if all_results:
                with st.expander("Algorithm Comparison", expanded=True):
                    st.subheader("Algorithm Comparison")
                    
                    # Create comparison dataframe
                    comparison_data = []
                    for result in all_results:
                        metrics = result.get('metrics', {})
                        if isinstance(metrics, dict):
                            row = {
                                'Algorithm': result['method'],
                                'Clusters Found': result.get('n_clusters', 'N/A')
                            }
                            if metrics.get('silhouette') is not None:
                                row['Silhouette Score'] = metrics['silhouette']
                            if metrics.get('Davies-Bouldin') is not None:
                                row['Davies-Bouldin'] = metrics['Davies-Bouldin']
                            if metrics.get('Calinski-Harabasz') is not None:
                                row['Calinski-Harabasz Score'] = metrics['Calinski-Harabasz']
                            comparison_data.append(row)
                        else:
                            st.warning(f"Skipping {result['method']} due to invalid or missing metrics.")
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.dataframe(comparison_df.style.format({
                                'Silhouette Score': '{:.4f}',
                                'Davies-Bouldin': '{:.4f}',
                                'Calinski-Harabasz Score': '{:.4f}'
                            }), height=400)
                        with col2:
                            try:
                                fig = plot_algorithm_metrics(all_results)
                                # Save comparison plot
                                comparison_plot_path = os.path.join(STATIC_DIR, "comparison_metrics.png")
                                plt.savefig(comparison_plot_path)
                                st.image(comparison_plot_path)
                                plt.close(fig)
                            except Exception as e:
                                st.error(f"Error plotting metrics comparison: {str(e)}")
                        
                        st.download_button(
                            label="Download Comparison CSV",
                            data=comparison_df.to_csv(index=False),
                            file_name="clustering_comparison.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No valid comparison data available.")

                # Cluster characteristics section
                with st.expander("Cluster Characteristics", expanded=False):
                    st.subheader("Cluster Characteristics")
                    try:
                        # Find the first valid result with labels
                        valid_results = [(r, lbl) for r, lbl in zip(all_results, all_labels) 
                                     if lbl is not None and len(np.unique(lbl)) > 1]
                        if valid_results:
                            # Get the best result based on silhouette score
                            best_result, best_labels = max(valid_results, 
                                                        key=lambda x: x[0]['metrics'].get('silhouette', -1) 
                                                        if isinstance(x[0].get('metrics', {}), dict) 
                                                        else -1)
                            fig = plot_cluster_characteristics(X_scaled, best_labels, best_result['method'])
                            characteristics_path = os.path.join(STATIC_DIR, "cluster_characteristics.png")
                            plt.savefig(characteristics_path)
                            st.image(characteristics_path)
                            plt.close(fig)
                        else:
                            st.warning("No valid clusters found for characteristics analysis.")
                    except Exception as e:
                        st.error(f"Error generating cluster characteristics: {str(e)}")

                # Cluster distribution section
                with st.expander("Cluster Distribution", expanded=False):
                    st.subheader("Cluster Distribution")
                    try:
                        # Find the first valid result with labels
                        valid_results = [(r, lbl) for r, lbl in zip(all_results, all_labels) 
                                     if lbl is not None and len(np.unique(lbl)) > 1]
                        if valid_results:
                            # Get the best result based on silhouette score
                            best_result, best_labels = max(valid_results, 
                                                        key=lambda x: x[0]['metrics'].get('silhouette', -1) 
                                                        if isinstance(x[0].get('metrics', {}), dict) 
                                                        else -1)
                            fig = plot_cluster_distribution(best_labels, best_result['method'])
                            distribution_path = os.path.join(STATIC_DIR, "cluster_distribution.png")
                            plt.savefig(distribution_path)
                            st.image(distribution_path)
                            plt.close(fig)
                        else:
                            st.warning("No valid clusters found for distribution analysis.")
                    except Exception as e:
                        st.error(f"Error generating cluster distribution: {str(e)}")

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin.")

# Footer
st.markdown("---")
st.markdown("""
### Algorithm Versions
Each algorithm implements multiple versions:
- **K-means Variants**: Standard K-Means, Hierarchical Clustering, Bisecting K-Means
- **DBSCAN**: Conservative, Aggressive, Balanced, Dense parameter settings
- **Genetic Algorithm**: Small Population Fast, Balanced, Large Population Long, Custom
- **Ant Colony**: Various pheromone update strategies
- **PSO**: Quick, Balanced, Thorough, Exploratory configurations
- **K-Medoids**: Fast, Balanced, Thorough, Robust configurations
- **Fuzzy C-Means**: Quick, Balanced, Fine, Fuzzy parameter settings
""")