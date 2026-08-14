# Credit Risk Ensemble Learning

A machine learning project for credit-risk classification using ensemble learning techniques and the UCI German Credit dataset.

> 🚧 **Status: Under Active Development**

## Overview

This project focuses on building a reproducible machine learning workflow for credit-risk classification.

The current implementation covers:

- Data ingestion and validation
- Feature preprocessing
- Numerical feature scaling
- Categorical feature encoding
- Ensemble model training
- Hyperparameter tuning
- Model evaluation
- Statistical analysis
- Automated testing
- Experiment artifacts and reports

The project is being developed incrementally with an emphasis on clean structure, reproducibility, testing, and clear documentation.

## Models

The project currently evaluates the following ensemble learning algorithms:

- **Random Forest**
- **Bagging**
- **AdaBoost**

## Dataset

The project uses the **German Credit dataset**.

The dataset is processed through the project's data ingestion and preprocessing pipeline before being used for model training and evaluation.

Local dataset files are excluded from version control to keep the repository lightweight and reproducible.

## Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Loading & Validation
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Model Training & Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Statistical Analysis
   ↓
Reports & Visualizations
```

### 1. Data Ingestion

Loads and validates the German Credit dataset using the project's data-loading module.

### 2. Data Preprocessing

Prepares the dataset for machine learning by:

- Separating features and target
- Identifying numerical and categorical features
- Scaling numerical features
- Encoding categorical features

### 3. Model Training

Trains and tunes the ensemble learning models:

- Random Forest
- Bagging
- AdaBoost

### 4. Model Evaluation

Models are evaluated using classification metrics including:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

The project also generates evaluation visualizations such as ROC curves, Precision-Recall curves, and confusion matrices.

### 5. Statistical Analysis

Statistical hypothesis testing is used to compare model performance and determine whether observed differences between models are meaningful.

### 6. Testing

The project includes automated tests for important components of the machine learning pipeline.

The test suite covers:

- Configuration
- Data loading
- Preprocessing
- Evaluation
- Training pipeline

## Project Structure

```text
credit-risk-ensemble-learning/
│
├── assets/                    # Generated visualizations
├── data/                      # Local dataset files
│   ├── raw/                   # Original dataset
│   └── processed/             # Processed dataset
│
├── models/                    # Saved model artifacts
├── reports/                   # Evaluation reports and results
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Project configuration
│   ├── data_loader.py         # Dataset loading and validation
│   ├── preprocessing.py       # Feature preprocessing
│   ├── train.py               # Model training and tuning
│   ├── evaluate.py            # Model evaluation
│   └── hypothesis_testing.py  # Statistical model comparison
│
├── tests/
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_evaluate.py
│   ├── test_preprocessing.py
│   └── test_training.py
│
├── main.py                    # End-to-end ML pipeline entry point
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd credit-risk-ensemble-learning
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Complete ML Pipeline

Run the main pipeline from the project root:

```bash
python main.py
```

The pipeline performs:

1. Data loading
2. Data preprocessing
3. Model training and hyperparameter tuning
4. Model evaluation
5. Statistical hypothesis testing

Generated reports and visualizations are stored in the project's output directories.

### 4. Run the Test Suite

Run all automated tests:

```bash
python -m pytest tests/
```

To run a specific test file:

```bash
python -m pytest tests/test_preprocessing.py
```

## Reproducibility

The project is structured to support reproducible machine learning experiments through:

- Centralized configuration
- Consistent preprocessing
- Separate training and evaluation stages
- Automated testing
- Organized output artifacts
- Version-controlled source code

## Project Status

🚧 **Under Active Development**

Current development focuses on improving:

- Model training and validation
- Evaluation and statistical analysis
- Automated test coverage
- Experiment reproducibility
- Documentation
- Model inference and application capabilities

Additional improvements will be added as the project progresses.

## Tech Stack

**Programming Language**
- Python

**Machine Learning**
- Scikit-learn
- Ensemble Learning
- Classification
- Hyperparameter Tuning
- Model Evaluation
- Statistical Analysis

**Data Processing**
- Pandas
- NumPy

**Visualization**
- Matplotlib
- Seaborn

**Testing**
- Pytest

## License

This project is licensed under the MIT License.
