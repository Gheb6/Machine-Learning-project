"""
MONK Problems - K-Nearest Neighbors (KNN) Analysis
===================================================

This script performs a comprehensive analysis of KNN algorithms on the MONK dataset,
comparing different variants and hyperparameters to find the optimal configuration.

Author: Gabriele Righi & Edoardo Fiaschi
Date: November 2025
Course: Machine learning

MONK Dataset:
-------------
The MONK problems are a collection of three binary classification tasks designed to
test machine learning algorithms. Each problem has 6 categorical attributes (a1-a6)
and a binary target (0 or 1).

Experiments Conducted:
---------------------
1. Standard KNN with different distance metrics (Hamming, Euclidean, Manhattan, Minkowski)
2. Weighted KNN (uniform vs distance-based weighting)
3. Modified KNN (MKNN) with validity-based weighting
4. Radius-based KNN with varying radius values

Methodology:
-----------
- Proper train/validation/test split (80% train, 20% validation from training data)
- One-hot encoding for categorical features
- Grid search over hyperparameters using validation set
- Final evaluation on test set
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib as mpl
from matplotlib.collections import LineCollection
import seaborn as sns
from sklearn.metrics import pairwise_distances
import time  # Import time module for seed
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import os
import sys

# Configure plotting style for professional-looking figures
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Random seed based on current time for variability across runs
seed = int(time.time())
print(f"Random seed set to: {seed}")
# seed = 56


# ============================================================================
# 0️⃣ GOOGLE COLAB SETUP AND HELPER FUNCTIONS
# ============================================================================

def setup_google_drive():
    """
    Mount Google Drive if running in Google Colab and create output directory.
    
    Returns:
    --------
    output_dir : str
        Path to the output directory (local or Google Drive)
    is_colab : bool
        Whether running in Google Colab environment
    """
    try:
        # Check if running in Google Colab
        import google.colab
        is_colab = True
        
        print("🔵 Google Colab detected!")
        print("Mounting Google Drive...")
        
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Create output directory in Google Drive
        output_dir = '/content/drive/MyDrive/MONK_KNN_Results'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"✅ Google Drive mounted successfully!")
        print(f"📁 Output directory: {output_dir}")
        
        return output_dir, is_colab
        
    except ImportError:
        # Not running in Colab, use local directory
        is_colab = False
        output_dir = '.'
        print("💻 Running locally - files will be saved in current directory")
        
        return output_dir, is_colab


def save_figure(fig, filename, output_dir='.'):
    """
    Save figure to specified directory.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename (without path)
    output_dir : str
        Directory to save the figure
    """
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {filepath}")


# ============================================================================
# 1️⃣ DATA LOADING AND PREPROCESSING
# ============================================================================

def load_monk_data(problem_num, validation_split=0.2, random_state=seed):
    """
    Load MONK dataset and split training data into train/validation sets.
    
    Parameters:
    -----------
    problem_num : int
        MONK problem number (1, 2, or 3)
    validation_split : float, default=0.2
        Fraction of training data to use for validation
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    X_train : pd.DataFrame
        Training features (80% of original training data)
    X_val : pd.DataFrame
        Validation features (20% of original training data)
    X_test : pd.DataFrame
        Test features (original test set)
    y_train : pd.Series
        Training labels
    y_val : pd.Series
        Validation labels
    y_test : pd.Series
        Test labels
    """
    # Define file paths for train and test data
    train_path = f"monk_dataset/monks-{problem_num}.train"
    test_path = f"monk_dataset/monks-{problem_num}.test"
    
    # Column names: 6 attributes + target + id
    cols = ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']
    
    # Load data (whitespace-separated, ignore ID column)
    train_df = pd.read_csv(train_path, sep=r'\s+', names=cols, usecols=range(7))
    test_df = pd.read_csv(test_path, sep=r'\s+', names=cols, usecols=range(7))
    
    # Separate features (X) and target (y) from original training data
    X_train_full = train_df.drop('target', axis=1)
    y_train_full = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Split training data into train (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=validation_split,
        random_state=random_state,
        stratify=y_train_full  # Maintain class balance
    )
    
    # Print dataset statistics for verification
    print(f"\n{'='*60}")
    print(f"MONK Problem {problem_num} - Data Split")
    print(f"{'='*60}")
    print(f"Training set: {len(X_train)} samples ({(1-validation_split)*100:.0f}% of original training data)")
    print(f"Validation set: {len(X_val)} samples ({validation_split*100:.0f}% of original training data)")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training class distribution: {y_train.value_counts().to_dict()}")
    print(f"Validation class distribution: {y_val.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_data(X_train, X_val, X_test):
    """
    Apply one-hot encoding to categorical features.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features (categorical)
    X_val : pd.DataFrame
        Validation features (categorical)
    X_test : pd.DataFrame
        Test features (categorical)
    
    Returns:
    --------
    X_train_enc : np.ndarray
        Encoded training features
    X_val_enc : np.ndarray
        Encoded validation features
    X_test_enc : np.ndarray
        Encoded test features
    encoder : OneHotEncoder
        Fitted encoder object
    """
    # Initialize encoder
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore'
    )
    
    # Fit encoder on training data and transform all sets
    X_train_enc = encoder.fit_transform(X_train)
    X_val_enc = encoder.transform(X_val)
    X_test_enc = encoder.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Features after one-hot encoding: {X_train_enc.shape[1]}")
    print(f"Encoding expanded features by {X_train_enc.shape[1] / X_train.shape[1]:.1f}x")
    
    return X_train_enc, X_val_enc, X_test_enc, encoder


# ============================================================================
# 2️⃣ KNN VARIANTS AND EVALUATION
# ============================================================================

def evaluate_standard_knn(X_train, y_train, X_val, y_val, k_values, metrics):
    """
    Evaluate standard KNN classifier with different distance metrics on validation set.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Encoded training features
    y_train : pd.Series
        Training labels
    X_val : np.ndarray
        Encoded validation features
    y_val : pd.Series
        Validation labels
    k_values : list
        List of k values to test
    metrics : list
        List of distance metrics to test
    
    Returns:
    --------
    results : dict
        Dictionary mapping metric names to accuracy lists
    """
    results = {metric: [] for metric in metrics}
    
    for metric in metrics:
        print(f"\n--- Testing metric: {metric} ---")
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train, y_train)
            
            y_pred = knn.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            results[metric].append(acc)
            
            print(f"k={k}, metric={metric}: Validation Accuracy = {acc:.4f}")
    
    return results


def evaluate_weighted_knn(X_train, y_train, X_val, y_val, k_values, weights):
    """
    Evaluate KNN with different weighting schemes on validation set.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Encoded training features
    y_train : pd.Series
        Training labels
    X_val : np.ndarray
        Encoded validation features
    y_val : pd.Series
        Validation labels
    k_values : list
        List of k values to test
    weights : list
        List of weighting schemes
    
    Returns:
    --------
    results : dict
        Dictionary mapping weight schemes to accuracy lists
    """
    results = {weight: [] for weight in weights}
    
    for weight in weights:
        print(f"\n--- Testing weight: {weight} ---")
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric='hamming', weights=weight)
            knn.fit(X_train, y_train)
            
            y_pred = knn.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            results[weight].append(acc)
            
            print(f"k={k}, weight={weight}: Validation Accuracy = {acc:.4f}")
    
    return results


def evaluate_mknn(X_train, y_train, X_val, y_val, k_values, k_validity=3):
    """
    Evaluate Modified KNN (MKNN) with validity-based weighting on validation set.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Encoded training features
    y_train : pd.Series
        Training labels
    X_val : np.ndarray
        Encoded validation features
    y_val : pd.Series
        Validation labels
    k_values : list
        List of k values to test
    k_validity : int, default=3
        Number of neighbors for validity calculation
    
    Returns:
    --------
    accuracies : list
        Validation accuracies for each k value
    """
    dist_train = pairwise_distances(X_train, X_train, metric='hamming')
    validity = []
    
    print(f"\n--- Computing validity scores (k={k_validity}) ---")
    for i in range(len(X_train)):
        idx = np.argsort(dist_train[i])[1:k_validity+1]
        same_class = np.sum(y_train.iloc[idx].values == y_train.iloc[i])
        validity.append(same_class / k_validity)
    
    validity = np.array(validity)
    print(f"Average validity score: {np.mean(validity):.4f}")
    
    accuracies = []
    print(f"\n--- Testing Modified KNN (validity k={k_validity}) ---")
    
    for k in k_values:
        y_pred = []
        
        for xv in X_val:
            dists = np.sum(np.abs(X_train - xv), axis=1) / X_train.shape[1]
            nn_idx = np.argsort(dists)[:k]
            
            eps = 1e-5
            weights = validity[nn_idx] / (dists[nn_idx] + eps)
            
            classes = y_train.iloc[nn_idx].values
            vote = {}
            for c, w in zip(classes, weights):
                vote[c] = vote.get(c, 0) + w
            
            y_pred.append(max(vote, key=vote.get))
        
        y_pred = np.array(y_pred)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
        print(f"k={k}: Validation Accuracy = {acc:.4f}")
    
    return accuracies


def evaluate_radius_knn(X_train, y_train, X_val, y_val, radius_values):
    """
    Evaluate Radius-based KNN classifier on validation set.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Encoded training features
    y_train : pd.Series
        Training labels
    X_val : np.ndarray
        Encoded validation features
    y_val : pd.Series
        Validation labels
    radius_values : list
        List of radius values to test
    
    Returns:
    --------
    valid_radii : list
        Radii that produced valid predictions
    accuracies : list
        Corresponding validation accuracies
    avg_neighbors : list
        Average number of neighbors found per sample
    """
    accuracies = []
    valid_radii = []
    avg_neighbors = []
    
    print(f"\n--- Testing Radius Neighbors ---")
    
    for r in radius_values:
        try:
            rknn = RadiusNeighborsClassifier(
                radius=r, 
                metric='hamming', 
                weights='uniform',
                outlier_label='most_frequent'
            )
            rknn.fit(X_train, y_train)
            
            # Get neighbors to check average count
            neighbors = rknn.radius_neighbors(X_val, return_distance=False)
            avg_n = np.mean([len(n) for n in neighbors])
            
            y_pred = rknn.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            
            accuracies.append(acc)
            valid_radii.append(r)
            avg_neighbors.append(avg_n)
            print(f"radius={r}: Validation Accuracy = {acc:.4f}, Avg neighbors = {avg_n:.1f}")
            
        except ValueError as e:
            print(f"radius={r}: FAILED - {e}")
    
    return valid_radii, accuracies, avg_neighbors


# ============================================================================
# 3️⃣ VISUALIZATION FUNCTIONS
# ============================================================================

def plot_comparison(k_values, results_dict, title, xlabel="k"):
    """
    Create line plot comparing different methods.
    
    This function generates a professional-looking plot showing how
    validation accuracy varies with k for different methods/metrics.
    
    Parameters:
    -----------
    k_values : list
        X-axis values (typically k values)
    results_dict : dict
        Dictionary mapping method names to accuracy lists
    title : str
        Plot title
    xlabel : str, default="k"
        X-axis label
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object (for saving)
    
    Features:
    ---------
    - Different colors and markers for each method
    - Grid for easier reading
    - Legend with method names
    - Y-axis from 0 to 1.05 (for accuracy)
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Define color scheme and marker styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # Plot each method
    for idx, (label, accuracies) in enumerate(results_dict.items()):
        ax.plot(k_values, accuracies, 
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2, markersize=8,
                label=label, alpha=0.8)
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig


def plot_heatmap_comparison(k_values, metrics, results, title):
    """
    Create heatmap showing accuracy for all (k, metric) combinations.
    
    Heatmaps provide an intuitive way to see which combinations work best:
    - Rows = different metrics
    - Columns = different k values
    - Color = validation accuracy (red=low, green=high)
    
    Parameters:
    -----------
    k_values : list
        K values (columns)
    metrics : list
        Metric names (rows)
    results : dict
        Dictionary mapping metrics to accuracy lists
    title : str
        Plot title
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object (for saving)
    
    Notes:
    ------
    - Uses 'RdYlGn' colormap (Red-Yellow-Green)
    - Annotations show exact accuracy values
    - Useful for identifying optimal hyperparameter regions
    """
    # Convert results to 2D array
    data = np.array([results[m] for m in metrics])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=k_values, yticklabels=metrics,
                vmin=0, vmax=1, ax=ax, 
                cbar_kws={'label': 'Test Accuracy'})
    
    ax.set_xlabel('k (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance Metric', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_pred_proba, title, save_path=None):
    """
    Plot ROC curve for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    auc_score : float
        Area Under the Curve (AUC) score
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add AUC text box
    textstr = f'AUC = {roc_auc:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.2, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, roc_auc


# ============================================================================
# 4️⃣ FINAL MODEL EVALUATION
# ============================================================================

def test_best_model(X_train, y_train, X_val, y_val, X_test, y_test, best_params, model_type='knn', problem_num=1, output_dir='.'):
    """
    Train and evaluate the best model on test set.
    
    This function:
    1. Re-trains on combined train+validation data
    2. Evaluates on test set (never seen during tuning)
    3. Generates detailed performance metrics
    4. Plots ROC curve
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    best_params : dict
        Best hyperparameters found during validation
    model_type : str, default='knn'
        Type of model: 'knn' or 'radius'
    problem_num : int, default=1
        MONK problem number for saving files
    output_dir : str, default='.'
        Directory to save output files
    
    Returns:
    --------
    test_acc : float
        Test set accuracy
    fig_cm : matplotlib.figure.Figure
        Confusion matrix figure
    fig_roc : matplotlib.figure.Figure
        ROC curve figure
    auc_score : float
        AUC score
    """
    print(f"\n{'='*60}")
    print("TESTING BEST MODEL ON TEST SET")
    print(f"{'='*60}")
    print(f"Model type: {model_type}")
    print(f"Best parameters: {best_params}")
    
    # Combine train and validation for final training
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # Encode all data
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_enc = encoder.fit_transform(X_train_full)
    X_test_enc = encoder.transform(X_test)
    
    # Train best model on full training set (train + validation)
    if model_type == 'radius':
        best_model = RadiusNeighborsClassifier(**best_params)
    else:
        best_model = KNeighborsClassifier(**best_params)
    
    best_model.fit(X_train_enc, y_train_full)
    
    # For radius classifier, report average neighbors on test set
    if model_type == 'radius':
        neighbors = best_model.radius_neighbors(X_test_enc, return_distance=False)
        avg_test_neighbors = np.mean([len(n) for n in neighbors])
        print(f"Average neighbors on test set: {avg_test_neighbors:.1f}")
    
    # Predict on test set
    y_pred = best_model.predict(X_test_enc)
    test_acc = accuracy_score(y_test, y_pred)
    
    # Get predicted probabilities for ROC curve
    y_pred_proba = best_model.predict_proba(X_test_enc)[:, 1]
    
    # Print detailed metrics
    print(f"\n📊 Test Accuracy: {test_acc:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix visualization
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_estimator(
        best_model, X_test_enc, y_test, 
        cmap='Blues', ax=ax
    )
    ax.set_title(f"Confusion Matrix - Test Set\nAccuracy: {test_acc:.4f}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Create ROC curve
    roc_save_path = os.path.join(output_dir, f'monk_{problem_num}_roc_curve.pdf')
    fig_roc, auc_score = plot_roc_curve(
        y_test, y_pred_proba,
        f"ROC Curve - MONK-{problem_num} (Test Set)",
        save_path=roc_save_path
    )
    
    print(f"\n📈 AUC Score: {auc_score:.4f}")
    
    return test_acc, fig_cm, fig_roc, auc_score


# ============================================================================
# 5️⃣ MAIN EXECUTION
# ============================================================================

def main(problem_num=1, random_state=None, output_dir='.'):
    """
    Main execution function - orchestrates entire analysis.
    
    Workflow:
    1. Load data and split into train (80%), validation (20%), and test
    2. Encode categorical features
    3. Run 4 experiments on validation set:
       a. Different distance metrics
       b. Different weighting schemes
       c. Modified KNN (MKNN)
       d. Radius-based KNN
    4. Compare all methods
    5. Select best model based on validation accuracy
    6. Retrain on train+validation and evaluate on test set
    7. Generate ROC curve
    
    Parameters:
    -----------
    problem_num : int
        MONK problem number (1, 2, or 3)
    random_state : int, optional
        Random seed for reproducibility
    output_dir : str, default='.'
        Directory to save output files
    """
    
    # Load data with validation split
    X_train, X_val, X_test, y_train, y_val, y_test = load_monk_data(
        problem_num, validation_split=0.2, random_state=random_state
    )
    
    # Encode features
    X_train_enc, X_val_enc, X_test_enc, encoder = encode_data(X_train, X_val, X_test)
    
    # Define hyperparameters
    k_values = list(range(1, 11))
    metrics = ['hamming', 'euclidean', 'manhattan', 'minkowski']
    weights = ['uniform', 'distance']
    radius_values = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    # EXPERIMENT 1: Different distance metrics
    print(f"\n{'#'*60}")
    print("EXPERIMENT 1: Standard KNN with Different Metrics")
    print(f"{'#'*60}")
    
    results_metrics = evaluate_standard_knn(X_train_enc, y_train, X_val_enc, y_val, 
                                           k_values, metrics)
    
    fig1 = plot_comparison(k_values, results_metrics, 
                          f"MONK-{problem_num}: KNN Performance by Distance Metric")
    save_figure(fig1, f'monk_{problem_num}_metrics_comparison.pdf', output_dir)
    plt.show()
    
    fig_heat1 = plot_heatmap_comparison(k_values, metrics, results_metrics,
                                       f"MONK-{problem_num}: Accuracy Heatmap - Distance Metrics")
    save_figure(fig_heat1, f'monk_{problem_num}_metrics_heatmap.pdf', output_dir)
    plt.show()
    
    # EXPERIMENT 2: Different weighting schemes
    print(f"\n{'#'*60}")
    print("EXPERIMENT 2: KNN with Different Weights")
    print(f"{'#'*60}")
    
    results_weights = evaluate_weighted_knn(X_train_enc, y_train, X_val_enc, y_val, 
                                           k_values, weights)
    
    fig2 = plot_comparison(k_values, results_weights, 
                          f"MONK-{problem_num}: KNN Performance by Weighting Scheme")
    save_figure(fig2, f'monk_{problem_num}_weights_comparison.pdf', output_dir)
    plt.show()
    
    # EXPERIMENT 3: Modified KNN
    print(f"\n{'#'*60}")
    print("EXPERIMENT 3: Modified KNN (MKNN)")
    print(f"{'#'*60}")
    
    results_mknn = evaluate_mknn(X_train_enc, y_train, X_val_enc, y_val, k_values)
    
    # EXPERIMENT 4: Radius KNN
    print(f"\n{'#'*60}")
    print("EXPERIMENT 4: Radius Neighbors")
    print(f"{'#'*60}")
    
    valid_radii, results_radius, avg_neighbors = evaluate_radius_knn(X_train_enc, y_train, 
                                                                      X_val_enc, y_val, radius_values)
    
    # Plot radius results if we have valid results
    if valid_radii:
        radius_dict = {f'Radius-{r:.2f}': [acc] for r, acc in zip(valid_radii, results_radius)}
        fig_radius = plot_comparison(valid_radii, {'Radius-KNN': results_radius}, 
                                    f"MONK-{problem_num}: Radius KNN Performance",
                                    xlabel="Radius")
        save_figure(fig_radius, f'monk_{problem_num}_radius_comparison.pdf', output_dir)
        plt.show()
    
    # FINAL COMPARISON
    print(f"\n{'#'*60}")
    print("FINAL COMPARISON OF ALL METHODS (VALIDATION SET)")
    print(f"{'#'*60}")
    
    all_results = {
        'KNN-Hamming': results_metrics['hamming'],
        'KNN-Euclidean': results_metrics['euclidean'],
        'Weighted-Distance': results_weights['distance'],
        'Modified-KNN': results_mknn
    }
    
    fig_final = plot_comparison(k_values, all_results, 
                               f"MONK-{problem_num}: Final Comparison of All Methods")
    save_figure(fig_final, f'monk_{problem_num}_final_comparison.pdf', output_dir)
    plt.show()
    
    # Find best model based on validation accuracy (including radius)
    best_val_acc = 0
    best_method = None
    best_k = None
    best_radius = None
    
    # Check k-based methods
    for method, accuracies in all_results.items():
        for k, acc in zip(k_values, accuracies):
            if acc > best_val_acc:
                best_val_acc = acc
                best_method = method
                best_k = k
                best_radius = None
    
    # Check radius-based method
    for r, acc, avg_n in zip(valid_radii, results_radius, avg_neighbors):
        if acc > best_val_acc:
            best_val_acc = acc
            best_method = 'Radius-KNN'
            best_k = None
            best_radius = r
            print(f"  (Radius {r} uses avg {avg_n:.1f} neighbors)")
    
    print(f"\n{'='*60}")
    print("BEST MODEL ON VALIDATION SET")
    print(f"{'='*60}")
    print(f"Method: {best_method}")
    if best_k is not None:
        print(f"k: {best_k}")
    if best_radius is not None:
        print(f"Radius: {best_radius}")
        idx = valid_radii.index(best_radius)
        print(f"Average neighbors: {avg_neighbors[idx]:.1f}")
    print(f"Validation Accuracy: {best_val_acc:.4f}")
    
    # Determine best parameters for final model
    if best_method == 'KNN-Hamming':
        best_params = {'n_neighbors': best_k, 'metric': 'hamming'}
        model_type = 'knn'
    elif best_method == 'KNN-Euclidean':
        best_params = {'n_neighbors': best_k, 'metric': 'euclidean'}
        model_type = 'knn'
    elif best_method == 'Weighted-Distance':
        best_params = {'n_neighbors': best_k, 'metric': 'hamming', 'weights': 'distance'}
        model_type = 'knn'
    elif best_method == 'Modified-KNN':
        best_params = {'n_neighbors': best_k, 'metric': 'hamming'}
        model_type = 'knn'
    else:  # Radius-KNN
        best_params = {'radius': best_radius, 'metric': 'hamming', 'outlier_label': 'most_frequent'}
        model_type = 'radius'
    
    # Test best model on test set
    test_acc, fig_cm, fig_roc, auc_score = test_best_model(
        X_train, y_train, X_val, y_val, 
        X_test, y_test, best_params, model_type, problem_num, output_dir
    )
    save_figure(fig_cm, f'monk_{problem_num}_confusion_matrix.pdf', output_dir)
    plt.show()
    
    # Show ROC curve
    plt.figure(fig_roc.number)
    plt.show()
    
    return {
        'best_method': best_method,
        'best_k': best_k,
        'best_radius': best_radius,
        'validation_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'auc_score': auc_score,
        'all_results': all_results,
        'radius_results': dict(zip(valid_radii, results_radius)) if valid_radii else {}
    }


if __name__ == "__main__":
    """
    Run analysis for all three MONK problems.
    """
    
    # Setup Google Drive if in Colab
    output_dir, is_colab = setup_google_drive()
    
    if is_colab:
        print("\n" + "="*70)
        print("📌 IMPORTANT: All results will be saved to:")
        print(f"   {output_dir}")
        print("="*70 + "\n")
    
    results_all = {}
    
    for problem_num in [1, 2, 3]:
        print(f"\n\n{'#'*70}")
        print(f"{'#'*70}")
        print(f"###          MONK PROBLEM {problem_num}          ###")
        print(f"{'#'*70}")
        print(f"{'#'*70}\n")
        
        results = main(problem_num, random_state=seed, output_dir=output_dir)
        results_all[problem_num] = results
    
    # Save summary to text file
    summary_path = os.path.join(output_dir, 'monk_results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SUMMARY: BEST MODELS FOR EACH MONK PROBLEM\n")
        f.write("="*70 + "\n\n")
        
        for prob, res in results_all.items():
            f.write(f"MONK-{prob}:\n")
            f.write(f"  Best Method: {res['best_method']}\n")
            if res['best_k'] is not None:
                f.write(f"  Optimal k: {res['best_k']}\n")
            if res['best_radius'] is not None:
                f.write(f"  Optimal radius: {res['best_radius']}\n")
            f.write(f"  Validation Accuracy: {res['validation_accuracy']:.4f}\n")
            f.write(f"  Test Accuracy: {res['test_accuracy']:.4f}\n")
            f.write(f"  AUC Score: {res['auc_score']:.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"Random seed used: {seed}\n")
        f.write("="*70 + "\n")
    
    print(f"\n💾 Summary saved to: {summary_path}")
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("SUMMARY: BEST MODELS FOR EACH MONK PROBLEM")
    print(f"{'='*70}")
    
    for prob, res in results_all.items():
        print(f"\nMONK-{prob}:")
        print(f"  Best Method: {res['best_method']}")
        if res['best_k'] is not None:
            print(f"  Optimal k: {res['best_k']}")
        if res['best_radius'] is not None:
            print(f"  Optimal radius: {res['best_radius']}")
        print(f"  Validation Accuracy: {res['validation_accuracy']:.4f}")
        print(f"  Test Accuracy: {res['test_accuracy']:.4f}")
        print(f"  AUC Score: {res['auc_score']:.4f}")
    
    print(f"\n{'='*70}")
    if is_colab:
        print(f"✅ Analysis complete! All files saved to Google Drive:")
        print(f"   {output_dir}")
    else:
        print("✅ Analysis complete! Check generated PDF files in current directory.")
    print(f"{'='*70}\n")