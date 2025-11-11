# Load and preprocess the Monk's Problems data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def one_hot_encode_monk_data(X):
    """
    Apply one-hot encoding to Monk's Problems features.
    Feature ranges: a1(1-3), a2(1-3), a3(1-2), a4(1-3), a5(1-4), a6(1-2)
    
    Args:
        X (np.array): Raw feature matrix
    
    Returns:
        np.array: One-hot encoded feature matrix
    """
    # Define the number of values for each feature
    feature_ranges = [3, 3, 2, 3, 4, 2]  # a1-a6
    
    encoded_features = []
    
    for i, num_values in enumerate(feature_ranges):
        # Create one-hot encoding for each feature
        feature_col = X[:, i]
        one_hot = np.zeros((len(feature_col), num_values))
        
        # Set the appropriate index to 1 (subtract 1 because values start at 1)
        for j, val in enumerate(feature_col):
            one_hot[j, val - 1] = 1
        
        encoded_features.append(one_hot)
    
    # Concatenate all encoded features
    return np.hstack(encoded_features)


def load_monk_data(problem_number, one_hot=True):
    """
    Load training and test data for a specific monk problem.
    
    Args:
        problem_number (int): Which monk problem to load (1, 2, or 3)
        one_hot (bool): Whether to apply one-hot encoding (default: True)
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) as numpy arrays
    """
    
    # File paths for the specific problem
    train_file = f'monk_dataset/monks-{problem_number}.train'
    test_file = f'monk_dataset/monks-{problem_number}.test'
    
    def parse_monk_file(filename):
        """Parse a monk dataset file and return features and labels."""
        data = []
        labels = []
        
        with open(filename, 'r') as f:
            for line in f:
                # Remove whitespace and split
                parts = line.strip().split()
                if len(parts) >= 7:  # Should have 7 elements (class + 6 attributes + id)
                    # First element is the class label (0 or 1)
                    labels.append(int(parts[0]))
                    # Next 6 elements are the features
                    features = [int(x) for x in parts[1:7]]
                    data.append(features)
        
        return np.array(data), np.array(labels)
    
    # Load training data
    print(f"Loading Monk Problem {problem_number} training data...")
    X_train, y_train = parse_monk_file(train_file)
    
    # Load test data
    print(f"Loading Monk Problem {problem_number} test data...")
    X_test, y_test = parse_monk_file(test_file)
    
    # Apply one-hot encoding if requested
    if one_hot:
        print("Applying one-hot encoding...")
        X_train = one_hot_encode_monk_data(X_train)
        X_test = one_hot_encode_monk_data(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Training labels distribution: {np.bincount(y_train)}")
    print(f"Test labels distribution: {np.bincount(y_test)}")
    
    return X_train, y_train, X_test, y_test


def analyze_monk_problem(problem_number):
    """
    Perform exploratory data analysis on a specific monk problem.
    
    Args:
        problem_number (int): Which monk problem to analyze (1, 2, or 3)
    """
    
    # Load the raw data (without one-hot encoding for visualization)
    X_train_raw, y_train, X_test_raw, y_test = load_monk_data(problem_number, one_hot=False)
    
    # Create feature names (a1 through a6 as per monk dataset convention)
    feature_names = [f'a{i}' for i in range(1, 7)]
    
    # Convert to DataFrame for easier analysis
    df_train = pd.DataFrame(X_train_raw, columns=feature_names)
    df_train['class'] = y_train
    
    print(f"\nAnalyzing Monk Problem {problem_number}")
    print("=" * 50)
    
    # Basic statistics
    print("Training data statistics:")
    print(df_train.describe())
    
    # Class distribution
    print(f"\nClass distribution in training set:")
    print(df_train['class'].value_counts())
    
    # Feature value ranges
    print(f"\nFeature value ranges:")
    for feature in feature_names:
        unique_values = sorted(df_train[feature].unique())
        print(f"{feature}: {unique_values}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Monk Problem {problem_number} - Feature Distribution by Class', fontsize=16)
    
    # Plot distribution of each feature by class
    for i, feature in enumerate(feature_names):
        row = i // 3
        col = i % 3
        
        # Get all possible values for this feature
        all_values = sorted(df_train[feature].unique())
        
        # Create grouped bar plot
        class_0_counts = df_train[df_train['class'] == 0][feature].value_counts().reindex(all_values, fill_value=0)
        class_1_counts = df_train[df_train['class'] == 1][feature].value_counts().reindex(all_values, fill_value=0)
        
        x = np.arange(len(all_values))
        width = 0.35
        
        axes[row, col].bar(x - width/2, class_0_counts.values, width, 
                          label='Class 0', alpha=0.8)
        axes[row, col].bar(x + width/2, class_1_counts.values, width, 
                          label='Class 1', alpha=0.8)
        
        axes[row, col].set_title(f'Feature {feature}')
        axes[row, col].set_xlabel('Feature Value')
        axes[row, col].set_ylabel('Count')
        axes[row, col].set_xticks(x)
        axes[row, col].set_xticklabels(all_values)
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'monk_problem_{problem_number}_analysis.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.show()
    
    return df_train


def visualize_one_hot_features(problem_number):
    """
    Visualize one-hot encoded features showing the 17 binary features.
    
    Args:
        problem_number (int): Which monk problem to visualize (1, 2, or 3)
    """
    # Load with one-hot encoding
    X_train, y_train, X_test, y_test = load_monk_data(problem_number, one_hot=True)
    
    # Feature names after one-hot encoding
    feature_names = []
    feature_ranges = [(1, 3), (1, 3), (1, 2), (1, 3), (1, 4), (1, 2)]  # a1-a6
    for i, (start, end) in enumerate(feature_ranges):
        for val in range(start, end + 1):
            feature_names.append(f'a{i+1}={val}')
    
    # Create DataFrame
    df = pd.DataFrame(X_train, columns=feature_names)
    df['class'] = y_train
    
    print(f"\nOne-Hot Encoded Features (17 binary features):")
    print(feature_names)
    print(f"\nFirst 5 samples:")
    print(df.head())
    
    # Calculate mean activation for each class
    class_0_mean = df[df['class'] == 0].drop('class', axis=1).mean()
    class_1_mean = df[df['class'] == 1].drop('class', axis=1).mean()
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    # Grouped bar chart comparing both classes
    axes[0].bar(x - width/2, class_0_mean.values, width, label='Class 0', alpha=0.8)
    axes[0].bar(x + width/2, class_1_mean.values, width, label='Class 1', alpha=0.8, color='orange')
    axes[0].set_title(f'Monk Problem {problem_number} - One-Hot Encoded Features (17 Binary Features)', fontsize=14)
    axes[0].set_xlabel('One-Hot Feature')
    axes[0].set_ylabel('Average Activation (Proportion)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='50% threshold')
    
    # Heatmap showing feature importance difference
    difference = class_1_mean.values - class_0_mean.values
    colors = ['red' if d < 0 else 'green' for d in difference]
    
    axes[1].bar(x, difference, color=colors, alpha=0.7)
    axes[1].set_title('Feature Importance: Class 1 - Class 0 (Positive = More in Class 1)', fontsize=14)
    axes[1].set_xlabel('One-Hot Feature')
    axes[1].set_ylabel('Difference in Activation')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'monk_problem_{problem_number}_onehot_analysis.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.show()
    
    return df


# Example usage
if __name__ == "__main__":
    # Load the first monk problem with one-hot encoding
    X_train, y_train, X_test, y_test = load_monk_data(1, one_hot=True)
    
    print("\nFirst 5 training samples (one-hot encoded):")
    print("Features shape:", X_train[:5].shape)
    print("Features:", X_train[:5])
    print("Labels:", y_train[:5])

    # Analyze all three monk problems
    for problem_num in [1, 2, 3]:
        df = analyze_monk_problem(problem_num)
        print(f"\nCompleted analysis for Monk Problem {problem_num}")
    
    # Visualize one-hot encoded features for all problems
    for problem_num in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"MONK PROBLEM {problem_num}")
        print(f"{'='*60}")
        df = visualize_one_hot_features(problem_num)