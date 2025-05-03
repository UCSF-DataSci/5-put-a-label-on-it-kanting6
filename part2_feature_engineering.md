# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 2: Time Series Features & Tree-Based Models

**Objective:** Extract basic time-series features from heart rate data, train Random Forest and XGBoost models, and compare their performance.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
```

## 2. Data Loading

Load the dataset.

```python
def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data with timestamp parsed as datetime
    """
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    return df
```

## 3. Feature Engineering

Implement `extract_rolling_features` to calculate rolling mean and standard deviation for the `heart_rate`.

```python
def extract_rolling_features(df, window_size_seconds):
    """
    Calculate rolling mean and standard deviation for heart rate.
    
    Args:
        df: DataFrame with timestamp and heart_rate columns
        window_size_seconds: Size of the rolling window in seconds
        
    Returns:
        DataFrame with added hr_rolling_mean and hr_rolling_std columns
    """
   def extract_rolling_features(df, window_size_seconds):
    """
    Calculate rolling mean and standard deviation for heart rate.
    
    Args:
        df: DataFrame with timestamp and heart_rate columns
        window_size_seconds: Size of the rolling window in seconds
        
    Returns:
        DataFrame with added hr_rolling_mean and hr_rolling_std columns
    """
    # 1. Sort by timestamp
    df_sorted = df.sort_values('timestamp')

    # 2. Set timestamp as index
    df_indexed = df_sorted.set_index('timestamp')

    # 3. Create rolling window and compute mean and std
    rolling_window = df_indexed['heart_rate'].rolling(window=f'{window_size_seconds}s')
    df_indexed['hr_rolling_mean'] = rolling_window.mean()
    df_indexed['hr_rolling_std'] = rolling_window.std()

    # 4. Reset index to bring timestamp back as a column
    df_result = df_indexed.reset_index()

    # 5. Handle missing values from rolling computations
    df_result = df_result.fillna(method='bfill')

    return df_result
```

## 4. Data Preparation

Implement `prepare_data_part2` using the newly engineered features.

```python
def prepare_data_part2(df_with_features, test_size=0.2, random_state=42):
    """
    Prepare data for modeling with time-series features.
    
    Args:
        df_with_features: DataFrame with original and rolling features
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # 1. Select feature columns (you can add more if desired)
    feature_cols = ['heart_rate', 'hr_rolling_mean', 'hr_rolling_std']
    X = df_with_features[feature_cols]
    
    # 2. Target variable
    y = df_with_features['disease_outcome']
    
    # 3. Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 4. Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    return X_train, X_test, y_train, y_test
```

## 5. Random Forest Model

Implement `train_random_forest`.

```python
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

```

## 6. XGBoost Model

Implement `train_xgboost`.

```python
def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
        learning_rate: Boosting learning rate
        max_depth: Maximum depth of a tree
        random_state: Random seed for reproducibility
        
    Returns:
        Trained XGBoost model
    """

     model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

```

## 7. Model Comparison

Calculate and compare AUC scores for both models.

```python
from sklearn.metrics import roc_auc_score

def compare_models(rf_model, xgb_model, X_test, y_test):
    """
    Compare the AUC scores of Random Forest and XGBoost models.
    
    Args:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        X_test: Test features
        y_test: True labels for the test set
        
    Returns:
        Dictionary with AUC scores
    """
    # 1. Predict probabilities
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # 2. Calculate AUC scores
    rf_auc = roc_auc_score(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    
    # 3. Compare
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    
    return {
        'Random Forest AUC': rf_auc,
        'XGBoost AUC': xgb_auc
    }

```

## 8. Save Results

Save the AUC scores to a text file.
```python

import os

def save_auc_scores(results_dict, filepath='results/results_part2.txt'):
    """
    Save AUC scores to a text file.
    
    Args:
        results_dict: Dictionary with model names and AUC scores
        filepath: Path to the output text file
    """
    # 1. Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 2. Format AUC scores as strings
    lines = [f"{model}: {auc:.4f}" for model, auc in results_dict.items()]

    # 3. Write to file
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))

```

## 9. Main Execution

Run the complete workflow.

```python
# Main execution

if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Extract rolling features
    window_size = 300  # 5 minutes in seconds
    df_with_features = extract_rolling_features(df, window_size)
    
    # 3. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part2(df_with_features)
    
    # 4. Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # 5. Calculate AUC scores
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    rf_auc = roc_auc_score(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    
    # 6. Save results
    results = {
        'Random Forest AUC': rf_auc,
        'XGBoost AUC': xgb_auc
    }
    save_auc_scores(results)
