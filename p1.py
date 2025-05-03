import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer


def load_data(file_path):
    """Load the synthetic health data with timestamp as datetime."""
    return pd.read_csv(file_path, parse_dates=['timestamp'])


def extract_rolling_features(df, window_size_seconds):
    """Calculate rolling mean and standard deviation for heart rate."""
    df_sorted = df.sort_values('timestamp')
    df_indexed = df_sorted.set_index('timestamp')
    rolling_window = df_indexed['heart_rate'].rolling(window=f'{window_size_seconds}s')
    df_indexed['hr_rolling_mean'] = rolling_window.mean()
    df_indexed['hr_rolling_std'] = rolling_window.std()
    df_result = df_indexed.reset_index()
    df_result = df_result.fillna(method='bfill')
    return df_result


def prepare_data_part2(df_with_features, test_size=0.2, random_state=42):
    """Prepare time-series feature set and split data."""
    feature_cols = ['heart_rate', 'hr_rolling_mean', 'hr_rolling_std']
    X = df_with_features[feature_cols]
    y = df_with_features['disease_outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    """Train an XGBoost classifier."""
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


def compare_models(rf_model, xgb_model, X_test, y_test):
    """Compare the AUC scores of Random Forest and XGBoost models."""
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    rf_auc = roc_auc_score(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)

    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")

    return {
        'Random Forest AUC': rf_auc,
        'XGBoost AUC': xgb_auc
    }


def save_auc_scores(results_dict, filepath='results/results_part2.txt'):
    """Save AUC scores to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lines = [f"{model}: {auc:.4f}" for model, auc in results_dict.items()]
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))


def main():
    # 1. Load data
    data_file = os.path.expanduser('~/Documents/5-put-a-label-on-it-kanting6/synthetic_health_data.csv')
    df = load_data(data_file)

    # 2. Extract rolling features
    window_size = 300  # 5 minutes in seconds
    df_with_features = extract_rolling_features(df, window_size)

    # 3. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part2(df_with_features)

    # 4. Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # 5. Compare AUC
    results = compare_models(rf_model, xgb_model, X_test, y_test)

    # 6. Save AUC scores
    save_auc_scores(results, filepath='results/results_part2.txt')


if __name__ == "__main__":
    main()
