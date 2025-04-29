import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the dataset and return the DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(df):
    """
    Explore the dataset with basic information and statistics.
    
    Args:
        df (pd.DataFrame): Dataset to explore
    """
    # Display the first few rows
    print("\n==== Dataset Overview ====")
    print(df.head())
    
    # Basic dataset information
    print("\n==== Dataset Information ====")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumn Types:")
    print(df.dtypes)
    
    # Statistical summary
    print("\n==== Statistical Summary ====")
    print(df.describe().round(2))
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n==== Missing Values ====")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found in the dataset.")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")

def visualize_features(df, save_dir):
    """
    Create visualizations for individual features.
    
    Args:
        df (pd.DataFrame): Dataset to visualize
        save_dir (str): Directory to save image files
    """
    plt.figure(figsize=(16, 12))
    plt.suptitle("Feature Distributions", fontsize=16)
    
    # Age distribution
    plt.subplot(2, 3, 1)
    sns.histplot(df['Age'], kde=True, bins=20, color='skyblue')
    plt.title('Age Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    # Gender distribution
    plt.subplot(2, 3, 2)
    gender_counts = df['Gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
            colors=['#ff9999','#66b3ff'], startangle=90)
    plt.title('Gender Distribution')
    
    # Annual Income distribution
    plt.subplot(2, 3, 3)
    sns.histplot(df['Annual Income (k$)'], kde=True, bins=20, color='lightgreen')
    plt.title('Annual Income Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    # Spending Score distribution
    plt.subplot(2, 3, 4)
    sns.histplot(df['Spending Score (1-100)'], kde=True, bins=20, color='salmon')
    plt.title('Spending Score Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    # Box plots for numerical features
    plt.subplot(2, 3, 5)
    numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    df_melted = pd.melt(df, value_vars=numerical_cols)
    sns.boxplot(x='variable', y='value', data=df_melted)
    plt.title('Feature Distributions (Box Plot)')
    plt.xticks(rotation=45)
    
    # Spending by gender
    plt.subplot(2, 3, 6)
    sns.boxplot(x='Gender', y='Spending Score (1-100)', data=df)
    plt.title('Spending Score by Gender')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure to the specified directory
    save_path = os.path.join(save_dir, 'feature_distributions.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

def plot_feature_relationships(df, save_dir):
    """
    Visualize relationships between different features.
    
    Args:
        df (pd.DataFrame): Dataset to visualize
        save_dir (str): Directory to save image files
    """
    plt.figure(figsize=(16, 12))
    plt.suptitle("Feature Relationships", fontsize=16)
    
    # Age vs Spending Score
    plt.subplot(2, 2, 1)
    plt.scatter(df['Age'], df['Spending Score (1-100)'], 
                c=df['Annual Income (k$)'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Annual Income (k$)')
    plt.title('Age vs Spending Score')
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')
    plt.grid(alpha=0.3)
    
    # Annual Income vs Spending Score
    plt.subplot(2, 2, 2)
    plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                c=df['Age'], cmap='plasma', alpha=0.7)
    plt.colorbar(label='Age')
    plt.title('Annual Income vs Spending Score')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.grid(alpha=0.3)
    
    # Correlation matrix
    plt.subplot(2, 2, 3)
    numerical_df = df.select_dtypes(include=[np.number])
    correlation = numerical_df.corr().round(2)
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    
    # Gender-based comparisons
    plt.subplot(2, 2, 4)
    sns.violinplot(x='Gender', y='Annual Income (k$)', data=df)
    plt.title('Annual Income by Gender')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure to the specified directory
    save_path = os.path.join(save_dir, 'feature_relationships.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

def segment_customers(df, save_dir):
    """
    Create visualizations to help identify potential customer segments.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        save_dir (str): Directory to save image files
    """
    plt.figure(figsize=(16, 12))
    plt.suptitle("Customer Segmentation Analysis", fontsize=16)
    
    # Age and spending across income groups
    plt.subplot(2, 2, 1)
    # Create income categories
    df['Income Category'] = pd.qcut(df['Annual Income (k$)'], 4, 
                            labels=['Low', 'Medium', 'High', 'Very High'])
    sns.boxplot(x='Income Category', y='Spending Score (1-100)', data=df)
    plt.title('Spending Score by Income Category')
    
    # Spending score categories
    plt.subplot(2, 2, 2)
    df['Spending Category'] = pd.qcut(df['Spending Score (1-100)'], 3, 
                              labels=['Low Spender', 'Average Spender', 'High Spender'])
    spending_counts = df['Spending Category'].value_counts()
    plt.bar(spending_counts.index, spending_counts.values, color=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Customer Spending Categories')
    plt.ylabel('Number of Customers')
    plt.grid(axis='y', alpha=0.3)
    
    # 3D scatter plot for segmentation
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(2, 2, 3, projection='3d')
    ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'],
               c=df['Spending Score (1-100)'], cmap='viridis', s=50, alpha=0.7)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score (1-100)')
    ax.set_title('3D Customer Segmentation')
    
    # Age categories vs spending
    plt.subplot(2, 2, 4)
    df['Age Group'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], 
                     labels=['Youth', 'Young Adult', 'Middle Age', 'Senior'])
    sns.barplot(x='Age Group', y='Spending Score (1-100)', hue='Gender', data=df)
    plt.title('Spending by Age Group and Gender')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure to the specified directory
    save_path = os.path.join(save_dir, 'customer_segmentation.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

def create_additional_visualizations(df, save_dir):
    """
    Create additional useful visualizations for customer analysis.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        save_dir (str): Directory to save image files
    """
    # 1. Pairplot of numerical features
    plt.figure(figsize=(12, 10))
    sns.pairplot(df, hue='Gender', vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    plt.suptitle("Pairwise Relationships Between Features", y=1.02, fontsize=16)
    
    save_path = os.path.join(save_dir, 'feature_pairplot.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()
    
    # 2. Income vs Spending with Gender distinction
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
                   hue='Gender', style='Gender', s=100, data=df)
    plt.title('Income vs Spending by Gender', fontsize=14)
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(save_dir, 'income_vs_spending_by_gender.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()
    
    # 3. Age distribution by gender
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Age', hue='Gender', multiple='stack', bins=20)
    plt.title('Age Distribution by Gender', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(save_dir, 'age_distribution_by_gender.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()
    
    # 4. KDE plots for spending score
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df, x='Spending Score (1-100)', hue='Gender', fill=True, common_norm=False)
    plt.title('Spending Score Density by Gender', fontsize=14)
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(save_dir, 'spending_density_by_gender.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

def preprocess_for_clustering(df):
    """
    Preprocess the data for clustering analysis.
    
    Args:
        df (pd.DataFrame): Dataset to preprocess
    
    Returns:
        tuple: (original features DataFrame, standardized features array)
    """
    # Select numerical features
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n==== Data prepared for clustering ====")
    print(f"Features: {X.columns.tolist()}")
    print(f"Data shape: {X_scaled.shape}")
    
    return X, X_scaled , scaler

def create_save_directory(save_dir):
    """
    Create the directory for saving images if it doesn't exist.
    
    Args:
        save_dir (str): Directory path
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        return True
    except Exception as e:
        print(f"Error creating directory: {e}")
        return False

def run_analysis(file_path, save_dir):
    """
    Run the complete analysis pipeline.
    
    Args:
        file_path (str): Path to the CSV file
        save_dir (str): Directory to save image files
    """
    # Create save directory if it doesn't exist
    if not create_save_directory(save_dir):
        print("Analysis aborted due to directory creation failure.")
        return
    
    # Load the data
    df = load_data(file_path)
    if df is None:
        return
    
    # Explore the data
    explore_data(df)
    
    # Create visualizations
    print("\nGenerating feature distribution visualizations...")
    visualize_features(df, save_dir)
    
    print("\nGenerating feature relationship visualizations...")
    plot_feature_relationships(df, save_dir)
    
    print("\nGenerating customer segmentation visualizations...")
    segment_customers(df, save_dir)
    
    print("\nGenerating additional visualizations...")
    create_additional_visualizations(df, save_dir)
    
    # Prepare data for clustering
    X, X_scaled,scaler = preprocess_for_clustering(df)
    
    print(f"\nAnalysis complete! All visualizations saved to: {save_dir}")
    return df, X, X_scaled,scaler