"""
File: config.py
Purpose: Hold all configuration constants, file paths, and hyperparameters in one central place.
Why it exists: Hardcoding values (like paths or parameters) directly inside code files makes the 
               code brittle and hard to update. Keeping them here allows us to change settings 
               without hunting through multiple files.
What it imports: os and pathlib to handle directory paths reliably across different operating systems.
Which files use it: Almost every file in the package (data generation, training, API) will import 
                    paths or settings from here.
Inputs: None directly, it just defines variables.
Outputs: Exposes variables to other modules.
Role in the pipeline: The backbone configuration that tells every script where to look for data, 
                      how to configure models, and where to save output.
"""

import os
from pathlib import Path

# ==========================================
# 1. PATH CONFIGURATIONS
# ==========================================

# We use pathlib to get an absolute path to the directory this file is in (the src/ folder),
# and then go up one level to get the root directory of the project (AmexPersonalization).
# Why? This ensures our paths work correctly no matter where the user runs the script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define directories for data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Define directories for models
MODELS_DIR = PROJECT_ROOT / "models"

# Define specific file paths to be used across the pipeline
RAW_DATA_PATH = RAW_DATA_DIR / "synthetic_logs.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"

# Paths to save the trained model and preprocessing tools
MODEL_SAVE_PATH = MODELS_DIR / "ranker_model.joblib"
ENCODER_SAVE_PATH = MODELS_DIR / "ordinal_encoder.joblib"

# ==========================================
# 2. DATA GENERATION PARAMETERS
# ==========================================
# We set parameters for our synthetic data generator so it's easy to adjust
# the scale of the dataset for testing vs. full runs.
NUM_USERS = 5000       # Number of unique users to simulate
NUM_OFFERS = 100       # Number of unique offers to simulate
NUM_LOGS = 60000       # Total number of interaction logs (rows) to generate
RANDOM_SEED = 42       # Random seed ensures our fake data is reproducible every time we run it

# ==========================================
# 3. FEATURE ENGINEERING & PREPROCESSING 
# ==========================================
# Defining which features fall into which category helps automate preprocessing.
CATEGORICAL_FEATURES = ["device_type", "offer_category", "user_segment"]

# Numerical features are continuous numbers
NUMERICAL_FEATURES = [
    "previous_clicks", 
    "previous_impressions", 
    "time_since_last_activity", 
    "transaction_count",
    "user_hist_ctr",        # Engineered feature: User's historical Click-Through Rate
    "offer_hist_ctr",       # Engineered feature: Offer's historical Click-Through Rate
    "activity_frequency",   # Engineered feature: How often user is active
    "offer_popularity"      # Engineered feature: How often offer was clicked globally
]

# The column we want to predict
TARGET_COL = "click"

# The column used to group data (ranking is done per-user)
GROUP_COL = "user_id"

# ==========================================
# 4. MODEL HYPERPARAMETERS
# ==========================================
# We use Optuna for tuning, but these are the limits we will search within
OPTUNA_TRIALS = 3            # Keeping it small for fast execution (a real project might use 50-100)
XGB_EARLY_STOPPING_ROUNDS = 10 # Stop training if validation score doesn't improve for 10 rounds

# ==========================================
# 5. EVALUATION METRICS
# ==========================================
K_FOR_MAP = 7 # We will calculate Mean Average Precision at K=7 (MAP@7)

