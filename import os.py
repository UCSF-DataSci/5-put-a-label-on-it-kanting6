import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


def load_data(file_path):
    """Load the synthetic health data from a CSV file."""
    return pd.read_csv(file_path)


def encode_categorical_features(df, column_to_encode='smoker_status'):
    """One-hot encode the specified categorical column, version-safe."""
    major, minor = map(int, sklearn.__version__.split(".")[:2])
    if major > 1 or (major == 1 and minor >= 2):
        encoder = OneHotEncoder(sparse_output=False, drop='first')
    else:
        encoder = OneHotEncoder(sparse=False, drop='first')

    encoded_array = encoder.fit_transform(df[[column_to_encode]])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out([column_to_encode]),
        index=df.index
    )

    df_encoded = df.drop(columns=[column_to_encode])
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    return df_encoded


def prepare_data_part3(df, test_size=0.2, random_state=42):
    """Prepare data with encoding, feature selection, and splitting."""
    df_encoded = encode_categorical_features(df)

    df_encoded = df_encoded.dropna()

    # Drop non-numeric columns like timestamp
    non_numeric_cols = df_encoded.select_dtypes(include=['object', 'datetime']).columns
    df_encoded = df_encoded.drop(columns=non_numeric_cols, errors='ignore')

    X = df_encoded.drop(columns=['disease_outcome'])
    y = df_encoded['disease_outcome']

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to balance the classes."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    return model


def calculate_evaluation_metrics(model, X_test, y_test):
    """Calculate standard classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def save_results(metrics, filename='results/results_part3.txt'):
    """Save metrics to a text file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    lines = []
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            lines.append(f"{key}:\n{value}")
        else:
            lines.append(f"{key}: {value:.4f}")
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


def load_part1_metrics(filepath='results/results_part1.txt'):
    """Load Part 1 metrics from a text file into a dictionary."""
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if ':' in line and not line.startswith("confusion_matrix"):
                    key, val = line.strip().split(':')
                    metrics[key.strip()] = float(val.strip())
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    return metrics


def compare_models(part1_metrics, part3_metrics):
    """Compute % improvement for metrics from Part 1 to Part 3."""
    improvement = {}
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for key in keys:
        old = part1_metrics.get(key, 0)
        new = part3_metrics.get(key, 0)
        if old == 0:
            improvement[key] = float('nan')
        else:
            improvement[key] = 100 * (new - old) / abs(old)
    return improvement


def main():
    # Load data
    data_file = os.path.expanduser('~/Documents/5-put-a-label-on-it-kanting6/synthetic_health_data.csv')
    df = load_data(data_file)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part3(df)

    # Apply SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # Train model
    model = train_logistic_regression(X_train_balanced, y_train_balanced)

    # Evaluate
    metrics = calculate_evaluation_metrics(model, X_test, y_test)

    # Print metrics
    print("\nEvaluation Metrics (Part 3 - Balanced):")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value:.4f}")

    # Save metrics
    save_results(metrics)

    # Compare with Part 1
    part1_metrics = load_part1_metrics()
    if part1_metrics:
        improvements = compare_models(part1_metrics, metrics)
        print("\nModel Comparison (% Improvement over Part 1):")
        for key, val in improvements.items():
            print(f"{key}: {val:.2f}%")
    else:
        print("\nNote: Could not load Part 1 results for comparison.")


if __name__ == "__main__":
    main()
