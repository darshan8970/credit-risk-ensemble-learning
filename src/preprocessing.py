import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    MODELS_DIR
)

def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for numerical scaling & categorical one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor

def prepare_data(df: pd.DataFrame, test_size: float = 0.20):
    """
    Splits dataset into stratified Train and Test sets (80:20),
    fits preprocessing pipeline on train data, and transforms both sets.
    
    Returns:
        X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Stratified Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERICAL_FEATURES + cat_feature_names

    # Save preprocessor artifact for inference / streamlit dashboard
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[INFO] Preprocessor fitted and saved to {preprocessor_path}")

    print(f"[INFO] Train samples: {X_train_proc.shape[0]}, Test samples: {X_test_proc.shape[0]}, Total Features: {X_train_proc.shape[1]}")
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names

if __name__ == "__main__":
    from src.data_loader import load_raw_dataset
    df = load_raw_dataset()
    prepare_data(df)
