import time

from src.data_loader import load_raw_dataset
from src.preprocessing import prepare_data
from src.train import train_and_tune_models
from src.evaluate import evaluate_models
from src.hypothesis_testing import run_hypothesis_tests


def main():
    start_time = time.time()

    print("=" * 70)
    print("       CREDIT RISK ENSEMBLE LEARNING PIPELINE")
    print("=" * 70)

    # Step 1: Data loading and preprocessing
    print("\n[STEP 1/4] Loading and preprocessing dataset...")
    df = load_raw_dataset()

    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    # Step 2: Model training and tuning
    print("\n[STEP 2/4] Training and tuning ensemble models...")
    trained_models, best_params = train_and_tune_models(X_train, y_train)

    # Step 3: Model evaluation
    print("\n[STEP 3/4] Evaluating trained models...")
    evaluate_models(
        trained_models,
        X_test,
        y_test,
        feature_names,
    )

    # Step 4: Statistical analysis
    print("\n[STEP 4/4] Running statistical hypothesis tests...")
    run_hypothesis_tests(
        trained_models,
        X_test,
        y_test,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 70)


if __name__ == "__main__":
    main()