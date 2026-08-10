import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.config import (
    PARAM_GRIDS,
    RANDOM_STATE,
    MODELS_DIR
)

def train_and_tune_models(X_train: np.ndarray, y_train: pd.Series):
    """
    Trains and tunes Random Forest, Bagging, and AdaBoost classifiers using Stratified 5-Fold GridSearchCV.
    
    Returns:
        dict: Trained model instances mapped by model name
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    trained_models = {}
    best_params_dict = {}

    # 1. Primary Model: Random Forest Classifier
    print("\n" + "="*60)
    print("[TRAINING] Tuning Random Forest Classifier (Primary Model)...")
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_grid = GridSearchCV(
        estimator=rf_base,
        param_grid=PARAM_GRIDS["Random Forest"],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    trained_models["Random Forest"] = rf_grid.best_estimator_
    best_params_dict["Random Forest"] = rf_grid.best_params_
    print(f"[BEST RF PARAMS] {rf_grid.best_params_}")
    print(f"[BEST RF CV ROC-AUC] {rf_grid.best_score_:.4f}")

    # 2. Baseline Model 1: Bagging Classifier (Bootstrap Aggregating with Decision Trees)
    print("\n" + "="*60)
    print("[TRAINING] Tuning Bagging Classifier (Baseline 1)...")
    bagging_base = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    )
    bagging_grid = GridSearchCV(
        estimator=bagging_base,
        param_grid=PARAM_GRIDS["Bagging"],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    bagging_grid.fit(X_train, y_train)
    trained_models["Bagging"] = bagging_grid.best_estimator_
    best_params_dict["Bagging"] = bagging_grid.best_params_
    print(f"[BEST BAGGING PARAMS] {bagging_grid.best_params_}")
    print(f"[BEST BAGGING CV ROC-AUC] {bagging_grid.best_score_:.4f}")

    # 3. Baseline Model 2: AdaBoost Classifier (Sequential Adaptive Boosting)
    print("\n" + "="*60)
    print("[TRAINING] Tuning AdaBoost Classifier (Baseline 2)...")
    adaboost_base = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        algorithm="SAMME"
    )
    adaboost_grid = GridSearchCV(
        estimator=adaboost_base,
        param_grid=PARAM_GRIDS["AdaBoost"],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    adaboost_grid.fit(X_train, y_train)
    trained_models["AdaBoost"] = adaboost_grid.best_estimator_
    best_params_dict["AdaBoost"] = adaboost_grid.best_params_
    print(f"[BEST ADABOOST PARAMS] {adaboost_grid.best_params_}")
    print(f"[BEST ADABOOST CV ROC-AUC] {adaboost_grid.best_score_:.4f}")

    # Save trained models
    for name, model in trained_models.items():
        filename = f"{name.lower().replace(' ', '_')}_model.joblib"
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, path)
        print(f"[SAVED] {name} saved to {path}")

    return trained_models, best_params_dict

if __name__ == "__main__":
    from src.data_loader import load_raw_dataset
    from src.preprocessing import prepare_data
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, _, _ = prepare_data(df)
    train_and_tune_models(X_train, y_train)
