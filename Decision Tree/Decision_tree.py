"""
Decision Tree Classifier for MONK Datasets

This script trains and evaluates Decision Tree classifiers on the three MONK datasets.

Key Features:
- One-hot encoding of categorical features (optional for comparison)
- Dataset-specific hyperparameter tuning via grid search
- 80/20 train/validation split for model selection
- Retraining on full training set for final evaluation
- Confusion matrix visualization
- Tree visualization
- Feature importance analysis
- Hyperparameter performance analysis
- Comparison across datasets

Author: Gabriele Righi & Edoardo Fiaschi
Date: November, 2025
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
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
    One-hot encoding converts each categorical value into a binary feature.
    
    Note: While decision trees can handle categorical features directly,
    one-hot encoding allows for fair comparison with SVM and can sometimes
    improve performance by making the feature space more explicit.
    
    Args:
        X_train (np.ndarray): Training feature matrix
        X_val (np.ndarray): Validation feature matrix
        X_test (np.ndarray): Test feature matrix
        
    Returns:
        tuple: (X_train_encoded, X_val_encoded, X_test_encoded, encoder, feature_names)
    """
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_train_encoded = encoder.fit_transform(X_train)
    X_val_encoded = encoder.transform(X_val)
    X_test_encoded = encoder.transform(X_test)
    
    # Get feature names for visualization
    feature_names = encoder.get_feature_names_out(['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
    
    return X_train_encoded, X_val_encoded, X_test_encoded, encoder, feature_names


def plot_confusion_matrix(y_true, y_pred, dataset_number):
    """
    Plot and save confusion matrix as PDF.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        dataset_number (int): MONK dataset number (1, 2, or 3)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    plt.title(f'Confusion Matrix - MONK-{dataset_number} Dataset (Test Set)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_dt_monk{dataset_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved as 'confusion_matrix_dt_monk{dataset_number}.pdf'")


def plot_decision_tree(tree_model, feature_names, dataset_number):
    """
    Visualize the decision tree structure.
    
    Args:
        tree_model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        dataset_number (int): MONK dataset number (1, 2, or 3)
    """
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model, 
              feature_names=feature_names,
              class_names=['0', '1'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title(f'Decision Tree Structure - MONK-{dataset_number}', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'decision_tree_structure_monk{dataset_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Decision tree structure saved as 'decision_tree_structure_monk{dataset_number}.pdf'")


def plot_feature_importance(tree_model, feature_names, dataset_number):
    """
    Plot feature importance from the decision tree.
    
    Args:
        tree_model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        dataset_number (int): MONK dataset number (1, 2, or 3)
    """
    importances = tree_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title(f'Top 10 Feature Importances - MONK-{dataset_number}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'feature_importance_monk{dataset_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance saved as 'feature_importance_monk{dataset_number}.pdf'")


def plot_depth_analysis(all_results, dataset_number):
    """
    Plot how tree depth affects performance.
    
    Args:
        all_results (list): List of dictionaries containing params and val_score
        dataset_number (int): MONK dataset number (1, 2, or 3)
    """
    # Group by max_depth
    depth_scores = {}
    for result in all_results:
        depth = result['params'].get('max_depth', 'None')
        depth_key = str(depth)
        if depth_key not in depth_scores:
            depth_scores[depth_key] = []
        depth_scores[depth_key].append(result['val_score'])
    
    # Sort depths
    sorted_depths = sorted(depth_scores.items(), key=lambda x: float('inf') if x[0] == 'None' else float(x[0]))
    depth_labels = [d[0] for d in sorted_depths]
    avg_scores = [np.mean(d[1]) for d in sorted_depths]
    std_scores = [np.std(d[1]) for d in sorted_depths]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(depth_labels)), avg_scores, yerr=std_scores, 
                 marker='o', linewidth=2, markersize=8, capsize=5)
    plt.xticks(range(len(depth_labels)), depth_labels)
    plt.xlabel('Max Depth', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title(f'Tree Depth vs Performance - MONK-{dataset_number}', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'depth_analysis_monk{dataset_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Depth analysis saved as 'depth_analysis_monk{dataset_number}.pdf'")


def plot_criterion_comparison(all_results, dataset_number):
    """
    Compare Gini vs Entropy criteria.
    
    Args:
        all_results (list): List of dictionaries containing params and val_score
        dataset_number (int): MONK dataset number (1, 2, or 3)
    """
    # Group by criterion
    criterion_best = {}
    for result in all_results:
        criterion = result['params']['criterion']
        score = result['val_score']
        if criterion not in criterion_best or score > criterion_best[criterion]:
            criterion_best[criterion] = score
    
    criteria = list(criterion_best.keys())
    scores = list(criterion_best.values())
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(criteria, scores, color=['#1f77b4', '#ff7f0e'])
    plt.xlabel('Splitting Criterion', fontsize=12)
    plt.ylabel('Best Validation Accuracy', fontsize=12)
    plt.title(f'Criterion Comparison - MONK-{dataset_number}', fontsize=14, fontweight='bold', pad=20)
    plt.ylim([0, 1.05])
    plt.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'criterion_comparison_monk{dataset_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Criterion comparison saved as 'criterion_comparison_monk{dataset_number}.pdf'")


def get_param_grid(dataset_number):
    """
    Get dataset-specific hyperparameter grid for grid search.
    
    Args:
        dataset_number (int): MONK dataset number (1, 2, or 3)
        
    Returns:
        list: List of parameter dictionaries
        
        DECISION TREE HYPERPARAMETERS EXPLAINED:
========================================

1. CRITERION (Splitting Quality Measure):
   - Determines how to measure the quality of a split
   - 'gini': Gini Impurity = 1 - Σ(p_i²)
     * Measures probability of incorrect classification
     * Range: [0, 0.5] for binary classification
     * 0 = pure node (all same class)
     * Fast to compute
   - 'entropy': Information Gain = -Σ(p_i * log₂(p_i))
     * Measures information content/uncertainty
     * Range: [0, 1] for binary classification
     * 0 = pure node, 1 = maximum uncertainty
     * Slightly slower but often finds better splits
   
   Example: Node with 100 samples, 60 class 0, 40 class 1
   - Gini = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 0.48
   - Entropy = -(0.6*log₂(0.6) + 0.4*log₂(0.4)) ≈ 0.97

2. MAX_DEPTH (Tree Complexity Control):
   - Maximum depth of the tree (root is depth 0)
   - None: Tree grows until all leaves are pure or meet other stopping criteria
   - Integer (e.g., 5): Tree stops growing at specified depth
   
   Effects:
   - Low depth (3-5): Simple model, prevents overfitting, faster predictions
   - High depth (15-20): Complex model, can overfit, captures intricate patterns
   - Unlimited (None): Very likely to overfit on training data
   
   Example: max_depth=3
   Root (depth 0) → Split 1 (depth 1) → Split 2 (depth 2) → Leaves (depth 3)

3. MIN_SAMPLES_SPLIT (Internal Node Splitting Threshold):
   - Minimum number of samples required to split an internal node
   - Default: 2 (split any node with 2+ samples)
   - Higher values: More conservative, prevents splitting small groups
   
   Effects:
   - Small (2-5): Aggressive splitting, detailed tree, prone to overfitting
   - Large (20-50): Conservative, simpler tree, better generalization
   
   Example: min_samples_split=10
   - Node with 15 samples → Can split
   - Node with 8 samples → Becomes a leaf (no split)
   
   Use case: In noisy datasets (like MONK-3), higher values prevent
   learning noise from small sample groups

4. MIN_SAMPLES_LEAF (Leaf Node Size Constraint):
   - Minimum number of samples required in a leaf node
   - Default: 1 (leaves can have single sample)
   - Higher values: Forces each leaf to represent more samples
   
   Effects:
   - Small (1-2): Very specific predictions, can memorize training data
   - Large (8-16): More general predictions, smoother decision boundaries
   
   Example: min_samples_leaf=5
   - A split that creates leaves with 10 and 7 samples → Valid
   - A split that creates leaves with 12 and 3 samples → Rejected
   
   Relationship with min_samples_split:
   - A node needs at least 2*min_samples_leaf samples to split
   - If min_samples_leaf=5, min_samples_split should be ≥10

OVERFITTING VS UNDERFITTING:
============================
Overfitting (Tree too complex):
- Symptoms: High training accuracy, low test accuracy
- Causes: max_depth=None, min_samples_split=2, min_samples_leaf=1
- Fix: Increase min_samples_split, min_samples_leaf, or decrease max_depth

Underfitting (Tree too simple):
- Symptoms: Low training and test accuracy
- Causes: Very small max_depth, very large min_samples_split/leaf
- Fix: Increase max_depth, decrease min_samples_split/leaf

DATASET-SPECIFIC STRATEGIES:
===========================
MONK-1 (Simple, linearly separable):
- Can use moderate complexity
- max_depth: 5-10 sufficient
- Low regularization needed

MONK-2 (Complex XOR-like):
- Needs higher complexity
- max_depth: 10-20 or None
- Allow smaller splits to capture complex patterns

MONK-3 (Noisy data):
- Needs strong regularization
- max_depth: 5-10
- min_samples_split: 10-30, min_samples_leaf: 4-16
- Prevents learning noise patterns

    """
    if dataset_number == 1:
        # MONK-1: Linearly separable, simple rule
        return [
            {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 4, 5],           # Lower max depth
                'min_samples_split': [5, 10, 15], # Higher threshold
                'min_samples_leaf': [2, 4, 6]     # Force larger leaves
            }
        ]
    elif dataset_number == 2:
        # MONK-2: Complex XOR-like problem - needs balance between complexity and regularization
        return [
            {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10],           # Reduced max depth
                'min_samples_split': [5, 10, 15, 20], # Higher thresholds to prevent overfitting
                'min_samples_leaf': [2, 4, 6, 8]      # Force larger leaves
            }
        ]
    else:  # MONK-3
        # MONK-3: Noisy data, needs regularization
        return [
            {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10, 15],
                'min_samples_split': [5, 10, 20, 30],
                'min_samples_leaf': [2, 4, 8, 16]
            }
        ]


def manual_grid_search(X_train, y_train, X_val, y_val, param_grid):
    """
    Perform manual grid search using hold-out validation.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation labels
        param_grid (list): List of parameter dictionaries
        
    Returns:
        tuple: (best_model, best_params, best_score, all_results)
    """
    from sklearn.model_selection import ParameterGrid
    
    best_score = 0
    best_params = None
    best_model = None
    all_results = []
    
    for param_dict in param_grid:
        for params in ParameterGrid(param_dict):
            # Train model with current parameters
            model = DecisionTreeClassifier(**params, random_state=42)
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


def train_and_evaluate_dt(dataset_number):
    """
    Train and evaluate Decision Tree on a specific MONK dataset.
    
    Training Process (Proper ML Workflow):
    ======================================
    
    PHASE 1: MODEL SELECTION (Using Validation Set)
    ------------------------------------------------
    1. Load monks-X.train file (full training data)
    2. Split into 80% train + 20% validation
    3. One-hot encode categorical features
    4. Perform grid search:
       - Train models on 80% portion
       - Evaluate on 20% validation portion
       - Select best hyperparameters based on validation accuracy
    
    WHY: Validation set acts as a proxy for test set during model selection.
         We never look at the test set during this phase!
    
    PHASE 2: FINAL MODEL TRAINING (Using All Training Data)
    --------------------------------------------------------
    5. Take the best hyperparameters from Phase 1
    6. Retrain model on FULL training data (train + validation combined)
       → This is the full monks-X.train file (100% of it)
    7. Evaluate final model on monks-X.test (untouched until now)
    
    WHY: After selecting hyperparameters, we want to use ALL available 
         training data to build the strongest possible model. The validation
         set has served its purpose (model selection) and can now be used
         for training the final model.
    
    IMPORTANT DISTINCTIONS:
    -----------------------
    - "Training set" in Phase 1 = 80% of monks-X.train
    - "Validation set" in Phase 1 = 20% of monks-X.train  
    - "Full training set" in Phase 2 = 100% of monks-X.train (train+val combined)
    - "Test set" = monks-X.test (ONLY used for final evaluation)
    
    Example with MONK-1 (124 training samples, 432 test samples):
    Phase 1: Split 124 → 99 train + 25 validation
             Try different hyperparameters on 99, evaluate on 25
             Best params: max_depth=5, min_samples_split=2, etc.
    
    Phase 2: Retrain with best params on all 124 samples
             Evaluate on 432 test samples
             Report final test accuracy
    
    This is standard practice in ML:
    - Validation set: for hyperparameter tuning
    - Test set: for final unbiased evaluation
    - After tuning, use all training data (including validation) for final model
    
    Args:
        dataset_number (int): MONK dataset number (1, 2, or 3)
        
    Returns:
        tuple: (dt_final, test_acc, best_params, all_results)
            - dt_final: Final model trained on full training set
            - test_acc: Accuracy on test set
            - best_params: Best hyperparameters from grid search
            - all_results: All grid search results for analysis
    """
    print(f"\n{'='*60}")
    print(f"MONK-{dataset_number} Dataset - Decision Tree")
    print(f"{'='*60}")
    
    # Load training and test data
    train_path = f'monks-{dataset_number}.train'
    test_path = f'monks-{dataset_number}.test'
    
    print(f"\nLoading data from files:")
    print(f"  Training file: {train_path}")
    print(f"  Test file: {test_path}")
    
    X_train_full, y_train_full = load_monk_data(train_path)
    X_test, y_test = load_monk_data(test_path)
    
    print(f"\nOriginal data sizes:")
    print(f"  Full training data (monks-{dataset_number}.train): {len(X_train_full)} samples")
    print(f"  Test data (monks-{dataset_number}.test): {len(X_test)} samples")
    
    # =========================================================================
    # PHASE 1: MODEL SELECTION USING HOLD-OUT VALIDATION
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: HYPERPARAMETER TUNING (Model Selection)")
    print(f"{'='*60}")
    
    # Split training data: 80% train, 20% validation
    # This split is ONLY for hyperparameter tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"\nSplit full training data for model selection:")
    print(f"  Training subset (80%): {len(X_train)} samples")
    print(f"  Validation subset (20%): {len(X_val)} samples")
    print(f"  → Total: {len(X_train) + len(X_val)} samples (same as full training data)")
    
    # One-hot encode categorical features
    print("\nApplying one-hot encoding to train/val split...")
    X_train_encoded, X_val_encoded, X_test_temp, encoder, feature_names = one_hot_encode_features(
        X_train, X_val, X_test
    )
    print(f"Feature dimension after encoding: {X_train_encoded.shape[1]}")
    
    # Get dataset-specific parameter grid
    param_grid = get_param_grid(dataset_number)
    
    # Perform manual grid search
    print("\nPerforming grid search...")
    print("  → Training on: 80% training subset")
    print("  → Validating on: 20% validation subset")
    print("  → Test set: NOT USED (saved for final evaluation)")
    
    best_model, best_params, best_score, all_results = manual_grid_search(
        X_train_encoded, y_train, X_val_encoded, y_val, param_grid
    )
    
    print(f"\n{'='*60}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"\nBest hyperparameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nBest validation accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
    
    # Evaluate best model on train and validation
    y_train_pred = best_model.predict(X_train_encoded)
    y_val_pred = best_model.predict(X_val_encoded)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"\nPerformance of best model (from grid search):")
    print(f"  Training subset accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation subset accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    overfitting_gap = train_acc - val_acc
    if overfitting_gap > 0.05:
        print(f"  ⚠ Overfitting detected: {overfitting_gap:.4f} gap between train and validation")
    elif overfitting_gap < -0.02:
        print(f"  ⚠ Unusual: validation better than training (might indicate small validation set)")
    else:
        print(f"  ✓ Good generalization: {overfitting_gap:.4f} gap between train and validation")
    
    # Display top 10 parameter combinations
    print("\nTop 10 hyperparameter combinations:")
    sorted_results = sorted(all_results, key=lambda x: x['val_score'], reverse=True)
    for rank, result in enumerate(sorted_results[:10], 1):
        params_str = ', '.join([f"{k}={v}" for k, v in result['params'].items()])
        print(f"  Rank {rank}: score={result['val_score']:.4f} | {params_str}")
    
    # =========================================================================
    # PHASE 2: FINAL MODEL TRAINING ON FULL TRAINING DATA
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: FINAL MODEL TRAINING (Retrain on Full Data)")
    print(f"{'='*60}")
    
    print("\nNow that we've selected the best hyperparameters,")
    print("we retrain the model on ALL available training data:")
    print(f"  → Using: Full training file (train + validation combined)")
    print(f"  → Size: {len(X_train_full)} samples (100% of monks-{dataset_number}.train)")
    print(f"  → Hyperparameters: {best_params}")
    
    # Re-encode with full training data
    encoder_final = OneHotEncoder(sparse_output=False, drop='first')
    X_train_full_encoded = encoder_final.fit_transform(X_train_full)
    X_test_encoded = encoder_final.transform(X_test)
    
    feature_names_final = encoder_final.get_feature_names_out(['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
    
    # Train final model with best hyperparameters on FULL training data
    print("\nTraining final model...")
    dt_final = DecisionTreeClassifier(**best_params, random_state=42)
    dt_final.fit(X_train_full_encoded, y_train_full)
    
    print(f"✓ Final model trained on {len(X_train_full)} samples")
    print(f"  Tree depth: {dt_final.get_depth()}")
    print(f"  Number of leaves: {dt_final.get_n_leaves()}")
    
    # =========================================================================
    # FINAL EVALUATION ON TEST SET
    # =========================================================================
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")
    
    # Predictions on full training set (to check for overfitting)
    y_train_full_pred = dt_final.predict(X_train_full_encoded)
    train_full_acc = accuracy_score(y_train_full, y_train_full_pred)
    
    # Predictions on test set (final unbiased evaluation)
    y_test_pred = dt_final.predict(X_test_encoded)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nFinal Model Performance:")
    print(f"  Full Training Set Accuracy: {train_full_acc:.4f} ({train_full_acc*100:.2f}%)")
    print(f"  Test Set Accuracy:          {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    generalization_gap = train_full_acc - test_acc
    print(f"  Generalization Gap:         {generalization_gap:.4f}")
    
    if generalization_gap > 0.10:
        print(f"  ⚠ Significant overfitting detected!")
    elif generalization_gap > 0.05:
        print(f"  ⚠ Moderate overfitting detected")
    elif generalization_gap < 0:
        print(f"  ✓ Test accuracy better than training (good generalization!)")
    else:
        print(f"  ✓ Good generalization")
    
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Class 0', 'Class 1']))
    
    print(f"\nTest Set Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"  True Negatives:  {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]} | True Positives:  {cm[1,1]}")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    plot_confusion_matrix(y_test, y_test_pred, dataset_number)
    plot_decision_tree(dt_final, feature_names_final, dataset_number)
    plot_feature_importance(dt_final, feature_names_final, dataset_number)
    plot_depth_analysis(all_results, dataset_number)
    plot_criterion_comparison(all_results, dataset_number)
    
    # Store metrics for saving to file
    metrics = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_full_acc': train_full_acc
    }
    
    return dt_final, test_acc, best_params, all_results, metrics


def plot_overall_comparison(all_dataset_results):
    """
    Plot comparison of all three MONK datasets performance.
    
    Args:
        all_dataset_results (dict): Dictionary with results for all datasets
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
    ax.set_title('Decision Tree Performance Comparison Across MONK Datasets', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('monk_overall_comparison_dt.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Overall comparison saved as 'monk_overall_comparison_dt.pdf'")


if __name__ == "__main__":
    """
    Main execution block.
    
    This script follows the standard machine learning workflow:
    
    FOR EACH MONK DATASET:
    1. Split training file → 80% train, 20% validation
    2. Grid search: find best hyperparameters using validation set
    3. Retrain: use best hyperparameters on FULL training file (train+val)
    4. Evaluate: test the final model on test file (never seen before)
    
    KEY PRINCIPLE: 
    - Validation set is for model selection (choosing hyperparameters)
    - After selection, we want to use ALL training data for final model
    - Test set is only touched once, for final unbiased evaluation
    """
    from datetime import datetime
    
    results = {}
    best_params_summary = {}
    all_dataset_results = {}
    all_metrics = {}
    
    print("="*60)
    print("DECISION TREE CLASSIFIER ON MONK DATASETS")
    print("="*60)
    print("\nThis script will:")
    print("1. Tune hyperparameters using 80/20 train/validation split")
    print("2. Retrain best model on full training data (train+validation)")
    print("3. Evaluate final model on test data")
    print("4. Generate comprehensive visualizations")
    
    for i in [1, 2, 3]:
        model, test_acc, best_params, grid_results, metrics = train_and_evaluate_dt(i)
        results[f'MONK-{i}'] = test_acc
        best_params_summary[f'MONK-{i}'] = best_params
        all_metrics[f'MONK-{i}'] = metrics
        
        # For overall comparison
        all_dataset_results[f'MONK-{i}'] = {
            'train_acc': metrics['train_acc'],
            'test_acc': test_acc
        }
    
    # Create overall comparison plot
    print("\n" + "="*60)
    print("GENERATING OVERALL COMPARISON")
    print("="*60)
    plot_overall_comparison(all_dataset_results)
    
    # Save results to text file
    results_filename = 'decision_tree_monk_results.txt'
    with open(results_filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Decision Tree MONK Dataset Results\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for dataset in ['MONK-1', 'MONK-2', 'MONK-3']:
            f.write(f"{dataset}\n")
            f.write("-"*40 + "\n")
            f.write(f"Training Accuracy:   {all_metrics[dataset]['train_acc']:.4f} ({all_metrics[dataset]['train_acc']*100:.2f}%)\n")
            f.write(f"Validation Accuracy: {all_metrics[dataset]['val_acc']:.4f} ({all_metrics[dataset]['val_acc']*100:.2f}%)\n")
            f.write(f"Test Accuracy:       {all_metrics[dataset]['test_acc']:.4f} ({all_metrics[dataset]['test_acc']*100:.2f}%)\n")
            f.write(f"Full Train Accuracy: {all_metrics[dataset]['train_full_acc']:.4f} ({all_metrics[dataset]['train_full_acc']*100:.2f}%)\n")
            f.write(f"Best Parameters:\n")
            for param, value in best_params_summary[dataset].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("Summary Table\n")
        f.write("="*60 + "\n")
        f.write(f"{'Dataset':<10} {'Train':<12} {'Validation':<12} {'Test':<12}\n")
        f.write("-"*46 + "\n")
        for dataset in ['MONK-1', 'MONK-2', 'MONK-3']:
            train = all_metrics[dataset]['train_acc']
            val = all_metrics[dataset]['val_acc']
            test = all_metrics[dataset]['test_acc']
            f.write(f"{dataset:<10} {train:.4f}       {val:.4f}       {test:.4f}\n")
        f.write("="*60 + "\n")
    
    print(f"\n✓ Results saved to '{results_filename}'")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS - DECISION TREE")
    print(f"{'='*60}")
    for dataset in ['MONK-1', 'MONK-2', 'MONK-3']:
        print(f"\n{dataset}:")
        print(f"  Training Accuracy:   {all_metrics[dataset]['train_acc']:.4f} ({all_metrics[dataset]['train_acc']*100:.2f}%)")
        print(f"  Validation Accuracy: {all_metrics[dataset]['val_acc']:.4f} ({all_metrics[dataset]['val_acc']*100:.2f}%)")
        print(f"  Test Accuracy:       {all_metrics[dataset]['test_acc']:.4f} ({all_metrics[dataset]['test_acc']*100:.2f}%)")
        print(f"  Best Parameters:")
        for param, value in best_params_summary[dataset].items():
            print(f"    {param}: {value}")
    
    print(f"\n{'='*60}")
    print("ALL RESULTS SAVED TO PDF AND TXT FILES")
    print(f"{'='*60}")