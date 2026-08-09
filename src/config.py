import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Ensure required directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, ASSETS_DIR]:
    os.makedirs(path, exist_ok=True)

# Dataset Configuration
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "german.data")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "german_credit_processed.csv")

# Random State
RANDOM_STATE = 42

# Column Names for German Credit Dataset (Statlog)
COLUMN_NAMES = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target"
]

NUMERICAL_FEATURES = [
    "duration",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents"
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]

TARGET_COLUMN = "target"

# Mapping readable category labels for UI and EDA (Expressed in Indian Rupees INR ₹)
CATEGORY_MAPPINGS = {
    "checking_status": {
        "A11": "Overdrawn / Deficit (< ₹0)",
        "A12": "Low Balance (₹0 to ₹10,000)",
        "A13": "High Balance (≥ ₹10,000 / Salary Assigned)",
        "A14": "No Checking Account"
    },
    "credit_history": {
        "A34": "Verified Excellent Credit History (Highest Score)",
        "A32": "Existing Loans Paid On Time (Clean Record)",
        "A30": "No Prior Credit History (First-Time Borrower)",
        "A33": "Restructured / Settled Past Accounts",
        "A31": "Frequent Delinquency / Past Credit Defaults"
    },
    "purpose": {
        "A40": "New Four-Wheeler (Car) Purchase",
        "A41": "Used Vehicle Purchase",
        "A42": "Furniture & Home Appliances",
        "A43": "Electronics / Television / Gadgets",
        "A44": "Domestic Appliances",
        "A45": "Home Repairs & Renovation",
        "A46": "Education & Higher Studies",
        "A47": "Vacation & Travel",
        "A48": "Skill Training & Certification",
        "A49": "Business Capital & Expansion",
        "A410": "Other Personal Expenses"
    },
    "savings_status": {
        "A61": "Low Savings (< ₹5,000)",
        "A62": "Moderate Savings (₹5,000 to ₹25,000)",
        "A63": "Good Savings (₹25,000 to ₹50,000)",
        "A64": "High Savings (≥ ₹50,000)",
        "A65": "No Savings Account / Unknown"
    },
    "employment_since": {
        "A71": "Currently Unemployed",
        "A72": "Employed Less than 1 Year",
        "A73": "Employed 1 to 4 Years",
        "A74": "Employed 4 to 7 Years",
        "A75": "Employed 7+ Years (Stable)"
    },
    "personal_status_sex": {
        "A91": "Male : Divorced / Separated",
        "A92": "Female : Divorced / Separated / Married",
        "A93": "Male : Single",
        "A94": "Male : Married / Widowed",
        "A95": "Female : Single"
    },
    "other_debtors": {
        "A101": "None (Sole Applicant)",
        "A102": "Co-Applicant Included",
        "A103": "Guarantor Guaranteed"
    },
    "property": {
        "A121": "Real Estate / Own Land / Flat",
        "A122": "Life Insurance / Fixed Deposit / Savings",
        "A123": "Car or Other Moveable Assets",
        "A124": "No Property / Rented Assets"
    },
    "other_installment_plans": {
        "A141": "Active Bank EMI Plan",
        "A142": "Store / Retail EMI Plan",
        "A143": "None (No Other Active EMIs)"
    },
    "housing": {
        "A151": "Rented Accommodation",
        "A152": "Self-Owned House",
        "A153": "Family-Provided / Free Housing"
    },
    "job": {
        "A171": "Unemployed / Daily Wage Worker",
        "A172": "Unskilled Resident Worker",
        "A173": "Skilled Professional / Salaried Employee",
        "A174": "Executive / Business Owner / Officer"
    },
    "telephone": {
        "A191": "No Registered Phone",
        "A192": "Yes, Registered Mobile/Phone"
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No"
    }
}

# Hyperparameter Search Grids for GridSearchCV
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced", None]
    },
    "Bagging": {
        "n_estimators": [30, 50, 100],
        "max_samples": [0.7, 0.85, 1.0],
        "max_features": [0.7, 0.85, 1.0]
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0]
    }
}
