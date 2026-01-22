import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# 1. REPRODUCIBILITY
# ---------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def calculate_mee_numpy(y_true, y_pred):
    """Calculates MEE using numpy (on original scale)."""
    return np.mean(np.sqrt(np.sum(np.square(y_pred - y_true), axis=1)))

def build_model(input_dim, output_dim, layers=[128, 128], lr=0.01, l2_reg=1e-5, dropout=0.0):
    """
    Enhanced network with dropout for better generalization:
    - ReLU activation (fastest computation)
    - MSE Loss (smoothest gradients for scaled data)
    - Dropout for regularization
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    for units in layers:
        model.add(Dense(units, activation='relu', 
                        kernel_initializer='he_uniform',
                        kernel_regularizer=l2(l2_reg)))
        if dropout > 0:
            model.add(Dropout(dropout))
            
    model.add(Dense(output_dim, activation='linear'))
    
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='mean_squared_error') # We monitor val_loss (MSE)
    return model

# 3. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------

filename = 'ML-CUP25-TR.csv'
data = pd.read_csv(filename, skiprows=7, header=None)

X = data.iloc[:, 1:13].values
y = data.iloc[:, 13:17].values

# 80% Development, 20% Test
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale Features
scaler_X = StandardScaler()
X_dev_scaled = scaler_X.fit_transform(X_dev)
X_test_scaled = scaler_X.transform(X_test)

# SCALE TARGETS (Crucial for Speed and Performance)
# This allows using higher learning rates and standard initialization
scaler_y = StandardScaler()
y_dev_scaled = scaler_y.fit_transform(y_dev)

# 4. GRID SEARCH (Final Push for MEE ~18-19)
# ---------------------------------------------------------

# BREAKTHROUGH: Best result 20.25 with [128,128,64,32], lr=0.005, l2=1e-4, dropout=0.08
# Strategy: Ultra-focused search around winning combination - strong L2 + minimal dropout
param_grid = [
    # Winning config variations - tune L2 and dropout precisely
    {'layers': [128, 128, 64, 32], 'lr': 0.005, 'l2': 1e-4, 'dropout': 0.05},
    {'layers': [128, 128, 64, 32], 'lr': 0.005, 'l2': 1e-4, 'dropout': 0.06},
    {'layers': [128, 128, 64, 32], 'lr': 0.005, 'l2': 1e-4, 'dropout': 0.08},  # Current best
    {'layers': [128, 128, 64, 32], 'lr': 0.005, 'l2': 8e-5, 'dropout': 0.08},
    {'layers': [128, 128, 64, 32], 'lr': 0.005, 'l2': 1.5e-4, 'dropout': 0.08},
    {'layers': [128, 128, 64, 32], 'lr': 0.005, 'l2': 2e-4, 'dropout': 0.06},
    
    # Alternative LR with best regularization
    {'layers': [128, 128, 64, 32], 'lr': 0.004, 'l2': 1e-4, 'dropout': 0.08},
    {'layers': [128, 128, 64, 32], 'lr': 0.006, 'l2': 1e-4, 'dropout': 0.08},
    {'layers': [128, 128, 64, 32], 'lr': 0.0045, 'l2': 1e-4, 'dropout': 0.07},
    
    # Slightly different architectures with same regularization strategy
    {'layers': [160, 128, 64, 32], 'lr': 0.005, 'l2': 1e-4, 'dropout': 0.08},
    {'layers': [128, 96, 64, 32], 'lr': 0.005, 'l2': 1e-4, 'dropout': 0.08},
    {'layers': [128, 128, 80, 40], 'lr': 0.005, 'l2': 1e-4, 'dropout': 0.08},
    {'layers': [128, 128, 64, 32], 'lr': 0.005, 'l2': 1.2e-4, 'dropout': 0.07},
    
    # No dropout - pure L2 regularization
    {'layers': [128, 128, 64, 32], 'lr': 0.005, 'l2': 2e-4, 'dropout': 0.0},
    {'layers': [128, 128, 64, 32], 'lr': 0.005, 'l2': 1.5e-4, 'dropout': 0.0},
    
    # Deeper with strong L2
    {'layers': [128, 128, 96, 64, 32], 'lr': 0.004, 'l2': 1e-4, 'dropout': 0.06},
]

best_score = float('inf')
best_params = None
best_avg_epochs = 0 

print(f"Starting 5-Fold Cross-Validation on Development Set ({len(X_dev)} samples)...")

for params in param_grid:
    print(f"\nTesting Configuration: {params}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_mee_scores = []
    fold_best_epochs = []
    
    for train_index, val_index in kf.split(X_dev_scaled):
        K.clear_session()
        
        X_train_fold, X_val_fold = X_dev_scaled[train_index], X_dev_scaled[val_index]
        y_train_fold_scaled, y_val_fold_scaled = y_dev_scaled[train_index], y_dev_scaled[val_index]
        y_val_fold_raw = y_dev[val_index] # Keep raw for real MEE calculation
        
        model = build_model(
            input_dim=X_train_fold.shape[1], 
            output_dim=y_train_fold_scaled.shape[1],
            layers=params['layers'], 
            lr=params['lr'],
            l2_reg=params['l2'],
            dropout=params['dropout']
        )
        
        # Enhanced Early Stopping for deeper networks
        # More patience for complex architectures
        early_stop = EarlyStopping(
            monitor='val_loss', mode='min', 
            patience=50, restore_best_weights=True, verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', mode='min', 
            factor=0.5, patience=15, min_lr=1e-7, verbose=0
        )
        
        history = model.fit(X_train_fold, y_train_fold_scaled, 
                            validation_data=(X_val_fold, y_val_fold_scaled),
                            epochs=400, batch_size=16, # More epochs for deeper networks
                            callbacks=[early_stop, reduce_lr], verbose=0)
        
        # Predict & Inverse Transform
        y_pred_scaled = model.predict(X_val_fold, verbose=0)
        y_pred_raw = scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate Real MEE
        mee = calculate_mee_numpy(y_val_fold_raw, y_pred_raw)
        fold_mee_scores.append(mee)
        
        # Track Best Epoch
        best_epoch_fold = np.argmin(history.history['val_loss']) + 1
        fold_best_epochs.append(best_epoch_fold)
    
    avg_mee = np.mean(fold_mee_scores)
    avg_epochs = int(np.mean(fold_best_epochs))
    
    print(f"  > Average CV MEE: {avg_mee:.4f}")
    print(f"  > Average Optimal Epochs: {avg_epochs}")
    
    if avg_mee < best_score:
        best_score = avg_mee
        best_params = params
        best_avg_epochs = avg_epochs

print("="*40)
print(f"Best Config: {best_params}")
print(f"Best CV MEE: {best_score:.4f}")
print(f"Optimal Training Epochs: {best_avg_epochs}")
print("="*40)

# 5. RETRAINING
# ---------------------------------------------------------
print(f"\nRetraining Best Model on FULL Development Set...")
print(f"Training for exactly {best_avg_epochs} epochs...")

K.clear_session()

final_model = build_model(
    input_dim=X_dev_scaled.shape[1],
    output_dim=y_dev_scaled.shape[1],
    layers=best_params['layers'],
    lr=best_params['lr'],
    l2_reg=best_params['l2'],
    dropout=best_params['dropout']
)

# Use ReduceLROnPlateau on training loss
reduce_lr_final = ReduceLROnPlateau(
    monitor='loss', mode='min', 
    factor=0.5, patience=15, min_lr=1e-7, verbose=0
)

final_model.fit(X_dev_scaled, y_dev_scaled, 
                epochs=best_avg_epochs,
                batch_size=16,
                callbacks=[reduce_lr_final],
                verbose=0)

# 6. EVALUATION
# ---------------------------------------------------------
print("\nEvaluating on Test Set...")

y_pred_test_scaled = final_model.predict(X_test_scaled, verbose=0)
y_pred_test_raw = scaler_y.inverse_transform(y_pred_test_scaled)

test_mee = calculate_mee_numpy(y_test, y_pred_test_raw)

print("-" * 30)
print(f"Final Test Set MEE: {test_mee:.4f}")
print("-" * 30)