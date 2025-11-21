"""
SVM Classifier for MONK Datasets

This script trains and evaluates Support Vector Machine (SVM) classifiers on the three MONK datasets.

Key Features:
- One-hot encoding of categorical features
- Dataset-specific hyperparameter tuning via grid search
- 80/20 train/validation split for model selection
- Retraining on full training set for final evaluation
- Confusion matrix visualization
- Hyperparameter performance heatmaps
- Comparison across datasets

Author: Gabriele Righi & Edoardo Fiaschi
Date: November, 2025
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def load_monk_data(filepath):
    """
    Load MONK dataset from file.
    
    The MONK datasets have a specific format:
    - First column: class label (0 or 1)
    - Next 6 columns: attributes a1-a6 (categorical values)
    - Additional columns are ignored (sample ID, etc.)
    
    Args:
        filepath (str): Path to the MONK dataset file
        
    Returns:
        tuple: (X, y) where X is the feature matrix (n_samples, 6) and 
               y is the label vector (n_samples,)
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            # Skip empty lines
            if line.strip():
                values = line.strip().split()
                # First value is the class (0 or 1), next 6 are attributes
                data.append([int(v) for v in values[:7]])
    
    df = pd.DataFrame(data, columns=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
    X = df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6']].values
    y = df['class'].values
    return X, y


def one_hot_encode_features(X_train, X_val, X_test):
    """
    One-hot encode categorical features.
    
    The MONK datasets contain categorical features with no ordinal relationship.
    One-hot encoding converts each categorical value into a binary feature,
    which is important for distance-based and linear models to avoid treating
    categorical values as having magnitude.
    
    Example: a1 ∈ {1, 2, 3} becomes three binary features: a1_2, a1_3 
    (a1_1 is dropped to avoid multicollinearity)
    
    Args:
        X_train (np.ndarray): Training feature matrix
        X_val (np.ndarray): Validation feature matrix
        X_test (np.ndarray): Test feature matrix
        
    Returns:
        tuple: (X_train_encoded, X_val_encoded, X_test_encoded, encoder)
               where encoded matrices have binary features and encoder can be
               reused for future transformations
    """
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_train_encoded = encoder.fit_transform(X_train)
    X_val_encoded = encoder.transform(X_val)
    X_test_encoded = encoder.transform(X_test)
    return X_train_encoded, X_val_encoded, X_test_encoded, encoder


def plot_confusion_matrix(y_true, y_pred, dataset_number):
    """
    Plot and save confusion matrix as PDF.
    
    A confusion matrix shows the counts of true positives, true negatives,
    false positives, and false negatives, providing insight into the types
    of errors the model makes.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        dataset_number (int): MONK dataset number (1, 2, or 3)
        
    Side Effects:
        Saves a PDF file named 'confusion_matrix_monk{dataset_number}.pdf'
        in the current directory
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    plt.title(f'Confusion Matrix - MONK-{dataset_number} Dataset (Test Set)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_monk{dataset_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved as 'confusion_matrix_monk{dataset_number}.pdf'")


def plot_hyperparameter_heatmap(all_results, dataset_number):
    """
    Plot heatmap of hyperparameter performance for RBF kernel.
    
    Shows how different C and gamma combinations affect validation accuracy,
    helping to visualize the hyperparameter search space and identify optimal regions.
    
    Args:
        all_results (list): List of dictionaries containing params and val_score
        dataset_number (int): MONK dataset number (1, 2, or 3)
        
    Side Effects:
        Saves a PDF file named 'hyperparameter_heatmap_monk{dataset_number}.pdf'
    """
    # Filter for RBF kernel results with numeric gamma
    rbf_results = [r for r in all_results if r['params'].get('kernel') == 'rbf' 
                   and isinstance(r['params'].get('gamma'), (int, float))]
    
    if not rbf_results:
        print("No RBF results with numeric gamma found for heatmap")
        return
    
    # Create pivot table
    data = []
    for result in rbf_results:
        data.append({
            'C': result['params']['C'],
            'gamma': result['params']['gamma'],
            'accuracy': result['val_score']
        })
    
    df = pd.DataFrame(data)
    pivot = df.pivot_table(values='accuracy', index='gamma', columns='C', aggfunc='mean')
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Validation Accuracy'})
    plt.title(f'RBF Kernel: C vs Gamma Performance - MONK-{dataset_number}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('C (Regularization Parameter)', fontsize=12)
    plt.ylabel('Gamma (Kernel Coefficient)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'hyperparameter_heatmap_monk{dataset_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hyperparameter heatmap saved as 'hyperparameter_heatmap_monk{dataset_number}.pdf'")


def plot_kernel_comparison(all_results, dataset_number):
    """
    Plot bar chart comparing performance of different kernel types.
    
    Shows the best validation accuracy achieved by each kernel type,
    helping to understand which kernel works best for the dataset.
    
    Args:
        all_results (list): List of dictionaries containing params and val_score
        dataset_number (int): MONK dataset number (1, 2, or 3)
        
    Side Effects:
        Saves a PDF file named 'kernel_comparison_monk{dataset_number}.pdf'
    """
    # Group by kernel and find best score
    kernel_best = {}
    for result in all_results:
        kernel = result['params']['kernel']
        score = result['val_score']
        if kernel not in kernel_best or score > kernel_best[kernel]:
            kernel_best[kernel] = score
    
    # Sort by score
    kernels = sorted(kernel_best.items(), key=lambda x: x[1], reverse=True)
    kernel_names = [k[0] for k in kernels]
    scores = [k[1] for k in kernels]
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(kernel_names, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(kernels)])
    plt.xlabel('Kernel Type', fontsize=12)
    plt.ylabel('Best Validation Accuracy', fontsize=12)
    plt.title(f'Kernel Comparison - MONK-{dataset_number} Dataset', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylim([0, 1.05])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'kernel_comparison_monk{dataset_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Kernel comparison saved as 'kernel_comparison_monk{dataset_number}.pdf'")


def plot_c_parameter_analysis(all_results, dataset_number):
    """
    Plot line chart showing how C parameter affects performance for each kernel.
    
    Helps visualize the effect of regularization strength on model performance.
    
    Args:
        all_results (list): List of dictionaries containing params and val_score
        dataset_number (int): MONK dataset number (1, 2, or 3)
        
    Side Effects:
        Saves a PDF file named 'c_parameter_analysis_monk{dataset_number}.pdf'
    """
    # Group with kernel and C
    kernel_c_scores = {}
    for result in all_results:
        kernel = result['params']['kernel']
        c = result['params']['C']
        score = result['val_score']
        
        if kernel not in kernel_c_scores:
            kernel_c_scores[kernel] = {}
        if c not in kernel_c_scores[kernel]:
            kernel_c_scores[kernel][c] = []
        kernel_c_scores[kernel][c].append(score)
    
    # Plot
    plt.figure(figsize=(12, 6))
    for kernel, c_scores in kernel_c_scores.items():
        c_values = sorted(c_scores.keys())
        avg_scores = [np.mean(c_scores[c]) for c in c_values]
        plt.plot(c_values, avg_scores, marker='o', label=kernel, linewidth=2, markersize=8)
    
    plt.xscale('log')
    plt.xlabel('C (Regularization Parameter)', fontsize=12)
    plt.ylabel('Average Validation Accuracy', fontsize=12)
    plt.title(f'C Parameter Impact on Performance - MONK-{dataset_number} Dataset', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'c_parameter_analysis_monk{dataset_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"C parameter analysis saved as 'c_parameter_analysis_monk{dataset_number}.pdf'")


def get_param_grid(dataset_number):
    """
    Get dataset-specific hyperparameter grid for grid search.
    
    Each MONK dataset has different characteristics that benefit from
    different SVM configurations:
    
    MONK-1: Linearly separable problem
        - Rule: (a1 == a2) OR (a5 == 1)
        - Best: Linear or RBF kernels with moderate C values
        
    MONK-2: Complex XOR-like problem
        - Rule: exactly two of {a1==1, a2==1, a3==1, a4==1, a5==1, a6==1}
        - Best: Polynomial or RBF kernels with higher complexity
        
    MONK-3: Noisy version of MONK-1
        - Rule: (a5 == 3 AND a4 == 1) OR (a5 != 4 AND a2 != 3) + 5% noise
        - Best: Regularized models (lower C) to handle noise
    
    Args:
        dataset_number (int): MONK dataset number (1, 2, or 3)
        
    Returns:
        list: List of parameter dictionaries for GridSearchCV.
              Each dictionary specifies a kernel type and its associated parameters.
    """
    if dataset_number == 1:
        # MONK-1: Linearly separable, try RBF and Poly with various settings
        return [
            {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
            {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]},
            {'kernel': ['poly'], 'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']},
        ]
    elif dataset_number == 2:
        # MONK-2: Complex XOR-like, needs polynomial or RBF
        return [
            {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
            {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10]},
            {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5], 'gamma': ['scale', 'auto', 0.1, 1], 'coef0': [0, 1, 10]},
        ]
    else:  # MONK-3
        # MONK-3: Noisy data, linear or mild non-linear
        return [
            {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]},
            {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1, 1]},
            {'kernel': ['poly'], 'C': [0.1, 1, 10, 100], 'degree': [2, 3], 'gamma': ['scale', 'auto']},
        ]


def manual_grid_search(X_train, y_train, X_val, y_val, param_grid):
    """
    Perform manual grid search using hold-out validation (80/20 split).
    
    This function evaluates all hyperparameter combinations on the
    validation set and returns the best model and parameters.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation labels
        param_grid (list): List of parameter dictionaries
        
    Returns:
        tuple: (best_model, best_params, best_score, all_results) where:
            - best_model: SVM model with best parameters
            - best_params: Dictionary of best hyperparameters
            - best_score: Best validation accuracy
            - all_results: List of all results for analysis
    """
    from sklearn.model_selection import ParameterGrid
    
    best_score = 0
    best_params = None
    best_model = None
    all_results = []
    
    # Flatten the list of parameter grids into individual parameter combinations
    for param_dict in param_grid:
        for params in ParameterGrid(param_dict):
            # Train model with current parameters
            model = SVC(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_score = model.score(X_val, y_val)
            
            # Store results
            all_results.append({
                'params': params,
                'val_score': val_score
            })
            
            # Update best model if necessary
            if val_score > best_score:
                best_score = val_score
                best_params = params
                best_model = model
    
    return best_model, best_params, best_score, all_results


def train_and_evaluate_svm(dataset_number):
    """
    Train and evaluate SVM on a specific MONK dataset.
    
    Training Process:
    1. Load training and test data
    2. Split training data into 80% train and 20% validation (hold-out)
    3. One-hot encode categorical features
    4. Standardize features (zero mean, unit variance)
    5. Perform grid search using hold-out validation to find best hyperparameters
    6. Evaluate best model on validation set
    7. Retrain on full training set (train + validation) with best hyperparameters
    8. Evaluate final model on test set
    9. Generate and save confusion matrix
    
    Args:
        dataset_number (int): MONK dataset number (1, 2, or 3)
        
    Returns:
        tuple: (svm_final, test_acc, best_params) where:
            - svm_final: Trained SVM model on full training set
            - test_acc: Test set accuracy (float)
            - best_params: Dictionary of best hyperparameters found
            
    Prints:
        - Data sizes
        - Grid search progress
        - Best parameters and validation accuracy
        - Top 10 parameter combinations
        - Training, validation, and test accuracies
        - Classification report and confusion matrix
    """
    print(f"\n{'='*60}")
    print(f"MONK-{dataset_number} Dataset")
    print(f"{'='*60}")
    
    # Load training and test data
    train_path = f'monk_dataset/monks-{dataset_number}.train'
    test_path = f'monk_dataset/monks-{dataset_number}.test'
    
    X_train_full, y_train_full = load_monk_data(train_path)
    X_test, y_test = load_monk_data(test_path)
    
    # Split training data: 80% train, 20% validation (hold-out)
    # Stratified split ensures class balance is maintained in both splits
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"\nData sizes:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # One-hot encode categorical features
    # This is crucial for MONK datasets as features are categorical with no ordinal relationship
    print("\nApplying one-hot encoding...")
    X_train_encoded, X_val_encoded, X_test_temp, encoder = one_hot_encode_features(X_train, X_val, X_test)
    print(f"Feature dimension after encoding: {X_train_encoded.shape[1]}")
    
    # Standardize features (mean=0, std=1)
    # While not strictly necessary for all kernels, standardization helps with:
    # - Numerical stability
    # - Faster convergence
    # - Fair comparison of C values across datasets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_val_scaled = scaler.transform(X_val_encoded)
    
    # Get dataset-specific parameter grid
    param_grid = get_param_grid(dataset_number)
    
    # Perform manual grid search using hold-out validation (80/20)
    print("\n--- Phase 1: Grid Search for Hyperparameter Tuning ---")
    print("Using hold-out validation (80% train, 20% validation)...")
    
    best_model, best_params, best_score, all_results = manual_grid_search(
        X_train_scaled, y_train, X_val_scaled, y_val, param_grid
    )
    
    print(f"\nBest parameters found: {best_params}")
    print(f"Best validation accuracy: {best_score:.4f}")
    
    # Evaluate best model on training and validation sets
    y_train_pred = best_model.predict(X_train_scaled)
    y_val_pred = best_model.predict(X_val_scaled)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"\nValidation Results with Best Model:")
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Display top 10 parameter combinations to understand what works well
    print("\nTop 10 parameter combinations:")
    sorted_results = sorted(all_results, key=lambda x: x['val_score'], reverse=True)
    for rank, result in enumerate(sorted_results[:10], 1):
        params_str = ', '.join([f"{k}={v}" for k, v in result['params'].items()])
        print(f"  Rank {rank}: {params_str}, score={result['val_score']:.4f}")
    
    # Retrain on full training data (train + validation) with best parameters
    # This maximizes the use of available training data for the final model
    print("\n--- Phase 2: Final Model Training on Full Training Set ---")
    
    # One-hot encode full training and test data
    # Important: Use a new encoder fitted on the full training set
    encoder_final = OneHotEncoder(sparse_output=False, drop='first')
    X_train_full_encoded = encoder_final.fit_transform(X_train_full)
    X_test_encoded = encoder_final.transform(X_test)
    
    # Standardize using full training set statistics
    scaler_final = StandardScaler()
    X_train_full_scaled = scaler_final.fit_transform(X_train_full_encoded)
    X_test_scaled = scaler_final.transform(X_test_encoded)
    
    # Train final model on all training data with best parameters
    svm_final = SVC(**best_params, random_state=42)
    svm_final.fit(X_train_full_scaled, y_train_full)
    
    # Final predictions
    y_train_full_pred = svm_final.predict(X_train_full_scaled)
    y_test_pred = svm_final.predict(X_test_scaled)
    
    # Calculate final accuracies
    train_full_acc = accuracy_score(y_train_full, y_train_full_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nFinal Results:")
    print(f"Full Training Accuracy: {train_full_acc:.4f} ({train_full_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Classification report provides precision, recall, and F1-score per class
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix shows distribution of predictions
    print(f"Test Set Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Generate all visualizations
    print("\n--- Generating Visualizations ---")
    plot_confusion_matrix(y_test, y_test_pred, dataset_number)
    plot_hyperparameter_heatmap(all_results, dataset_number)
    plot_kernel_comparison(all_results, dataset_number)
    plot_c_parameter_analysis(all_results, dataset_number)
    
    return svm_final, test_acc, best_params, all_results


def plot_overall_comparison(all_dataset_results):
    """
    Plot comparison of all three MONK datasets performance.
    
    Shows train vs test accuracy for each dataset to identify overfitting.
    
    Args:
        all_dataset_results (dict): Dictionary with results for all datasets
        
    Side Effects:
        Saves a PDF file named 'monk_overall_comparison.pdf'
    """
    datasets = ['MONK-1', 'MONK-2', 'MONK-3']
    train_accs = [all_dataset_results[d]['train_acc'] for d in datasets]
    test_accs = [all_dataset_results[d]['test_acc'] for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, train_accs, width, label='Training Accuracy', color='#2ca02c')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='#1f77b4')
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('SVM Performance Comparison Across MONK Datasets', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('monk_overall_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Overall comparison saved as 'monk_overall_comparison.pdf'")


if __name__ == "__main__":
    """
    Main execution block.
    
    Trains and evaluates SVM classifiers on all three MONK datasets,
    then prints a summary of results including test accuracies and
    best hyperparameters for each dataset.
    """
    # Train and evaluate on all three MONK datasets
    results = {}
    best_params_summary = {}
    all_dataset_results = {}
    
    for i in [1, 2, 3]:
        model, test_acc, best_params, grid_results = train_and_evaluate_svm(i)
        results[f'MONK-{i}'] = test_acc
        best_params_summary[f'MONK-{i}'] = best_params
        
        # Store for comparison plot
        # Get training accuracy from grid search results
        train_acc = max([r['val_score'] for r in grid_results])
        all_dataset_results[f'MONK-{i}'] = {
            'train_acc': train_acc,
            'test_acc': test_acc
        }
    
    # Create overall comparison plot
    print("\n--- Generating Overall Comparison ---")
    plot_overall_comparison(all_dataset_results)
    
    # Print summary of all results
    print(f"\n{'='*60}")
    print("Summary of Results")
    print(f"{'='*60}")
    for dataset in ['MONK-1', 'MONK-2', 'MONK-3']:
        print(f"\n{dataset}:")
        print(f"  Test Accuracy: {results[dataset]:.4f} ({results[dataset]*100:.2f}%)")
        print(f"  Best Parameters: {best_params_summary[dataset]}")

