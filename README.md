# Credit Risk Ensemble Learning

A machine learning project for credit-risk classification using ensemble learning techniques.

## Overview

This project explores machine learning approaches for predicting credit risk using the UCI German Credit dataset.

The project focuses on building a reproducible machine learning workflow covering:

- Data ingestion
- Data preprocessing
- Model training
- Model comparison
- Performance evaluation
- Statistical analysis
- Model interpretation
- Interactive prediction

## Models

The project evaluates multiple classification approaches, including:

- Random Forest
- Bagging
- AdaBoost

## Dataset

The project uses the German Credit dataset from the UCI Machine Learning Repository.

Dataset preparation and validation are implemented as part of the machine learning pipeline.

## Project Structure

The repository is organized into separate components for different stages of the machine learning workflow:

- `src/` – Core machine learning and data processing modules
- `data/` – Dataset files and processed data
- `models/` – Trained model artifacts
- `reports/` – Evaluation results and generated reports
- `assets/` – Supporting project assets
- `requirements.txt` – Python dependencies
- `LICENSE` – Project license

## ML Workflow

The project follows a structured machine learning workflow:

1. **Data Ingestion** – Loads the German Credit dataset.
2. **Data Preprocessing** – Separates features and target, scales numerical features, and encodes categorical features.
3. **Model Training** – Trains and tunes Random Forest, Bagging, and AdaBoost classifiers.
4. **Model Evaluation** – Compares model performance using classification metrics.
5. **Statistical Analysis** – Performs hypothesis testing to compare model performance.
6. **Results & Artifacts** – Generates evaluation results and stores project artifacts.

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt


## Status

🚧 **Under active development**

The project is being built incrementally with reproducibility and clear experiment tracking as key goals.

## License

This project is licensed under the MIT License.