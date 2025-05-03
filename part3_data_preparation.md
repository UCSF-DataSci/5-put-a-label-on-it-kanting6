# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 3: Practical Data Preparation

**Objective:** Handle categorical features using One-Hot Encoding and address class imbalance using SMOTE.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
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
        DataFrame containing the data
    """
    df = pd.read_csv(file_path)
    return df

```

## 3. Categorical Feature Encoding

Implement `encode_categorical_features` using `OneHotEncoder`.

```python
from sklearn.preprocessing import OneHotEncoder

def encode_categorical_features(df, column_to_encode='smoker_status'):
    """
    Encode a categorical column using OneHotEncoder.
    
    Args:
        df: Input DataFrame
        column_to_encode: Name of the categorical column to encode
        
    Returns:
        DataFrame with the categorical column replaced by one-hot encoded columns
    """
    # 1. Extract the column
    encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid multicollinearity
    encoded_array = encoder.fit_transform(df[[column_to_encode]])
    
    # 2. Create DataFrame with new column names
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out([column_to_encode]),
        index=df.index
    )
    
    # 3. Drop original column and concatenate encoded columns
    df_encoded = df.drop(columns=[column_to_encode])
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    return df_encoded

```

## 4. Data Preparation

Implement `prepare_data_part3` to handle the train/test split correctly.

```python
from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_data_part3(df, test_size=0.2, random_state=42):
    """
    Prepare data with categorical encoding, feature selection, and train/test split.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Step 1: Encode categorical features
    df_encoded = encode_categorical_features(df)

    # Step 2: Drop rows with missing values 
    df_encoded = df_encoded.dropna()

    # Step 3: Separate features and target
    X = df_encoded.drop(columns=['disease_outcome'])
    y = df_encoded['disease_outcome']

    # Step 4: Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

```

## 5. Handling Imbalanced Data

Implement `apply_smote` to oversample the minority class.

```python
def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to oversample the minority class.
    
    Args:
        X_train: Training features
        y_train: Training disease_outcome
        random_state: Random seed for reproducibility
        
    Returns:
        Resampled X_train and y_train with balanced classes
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
```

## 6. Model Training and Evaluation

Train a model on the SMOTE-resampled data and evaluate it.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training disease_outcome
        
    Returns:
        Trained logistic regression model
    """
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    return model

def calculate_evaluation_metrics(model, X_test, y_test):
    """
    Calculate classification evaluation metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test disease_outcome
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, auc, and confusion_matrix
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    return metrics

```

## 7. Save Results

Save the evaluation metrics to a text file.

```python
import os

def save_results(metrics, filename='results/results_part3.txt'):
    """
    Save evaluation metrics to a text file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        filename: Path to output file
    """
    # Step 1: Create 'results' directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Step 2: Format metrics as strings
    lines = []
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            lines.append(f"{key}:\n{value}\n")
        else:
            lines.append(f"{key}: {value:.4f}")

    # Step 3: Write to file
    with open(filename, 'w') as f:
        f.write("\n".join(lines))

```

## 8. Main Execution

Run the complete workflow.

```python
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data with categorical encoding
    X_train, X_test, y_train, y_test = prepare_data_part3(df)
    
    # 3. Apply SMOTE to balance the training data
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # 4. Train model on resampled data
    model = train_logistic_regression(X_train_resampled, y_train_resampled)
    
    # 5. Evaluate on original test set
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 6. Print metrics
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    # 7. Save results
    save_results(metrics)
    
    # 8. Load Part 1 results for comparison
    import json
    try:
        with open('results/results_part1.txt', 'r') as f:
            part1_metrics = json.load(f)
        
        # 9. Compare models
        comparison = compare_models(part1_metrics, metrics)
        print("\nModel Comparison (improvement percentages):")
        for metric, improvement in comparison.items():
            print(f"{metric}: {improvement:.2f}%")
    except FileNotFoundError:
        print("Part 1 results not found. Run part1_introduction.ipynb first.")
```

## 9. Compare Results

Implement a function to compare model performance between balanced and imbalanced data.

```python
def compare_models(part1_metrics, part3_metrics):
    """
    Calculate percentage improvement between models trained on imbalanced vs. balanced data.
    
    Args:
        part1_metrics: Dictionary containing evaluation metrics from Part 1 (imbalanced)
        part3_metrics: Dictionary containing evaluation metrics from Part 3 (balanced)
        
    Returns:
        Dictionary with metric names as keys and improvement percentages as values
    """
    improvement = {}
    
    # Metrics where higher is better
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    for metric in metrics_to_compare:
        old_val = part1_metrics.get(metric, 0)
        new_val = part3_metrics.get(metric, 0)
        
        if old_val == 0:
            improvement[metric] = float('nan')  # Avoid divide-by-zero
        else:
            percent_change = ((new_val - old_val) / abs(old_val)) * 100
            improvement[metric] = percent_change

    return improvement
