"""
File: preprocess.py
Purpose: Clean the data, handle missing values, encode text into numbers, handle class imbalance, 
         and split into Train/Validation sets.
Why it exists: ML models only understand numbers. They can't eat raw text like 'mobile_app'. Also, 
               if 98% of the data is 'no click', the model might just learn to guess 'no' every time. 
               Preprocessing solves these issues.
What it imports: scikit-learn tools, imbalanced-learn for SMOTE, config.
Which files use it: train.py.
Inputs: DataFrame with engineered features.
Outputs: X_train, y_train, X_val, y_val dataframes, along with fitted encoders.
Role in the pipeline: Step 3. Prepares the clean, balanced numerical matrix the model trains on.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from src import config
from src.utils import get_logger, save_object

logger = get_logger(__name__)

def preprocess_data(df: pd.DataFrame, is_training: bool = True):
    """
    Cleans data, encodes categorical features, and handles class imbalance.
    
    Why this exists:
    We must transform categorical strings to numbers (Encoding).
    If training, we also do Train/Validation Splitting and handle Class Imbalance (SMOTE).
    
    Inputs:
        df: The pandas DataFrame with engineered features.
        is_training (bool): If True, we fit new encoders and apply SMOTE. 
                            If False (during inference), we load existing encoders.
                            
    Outputs:
        If is_training: Returns (X_train, X_val, y_train, y_val, group_val)
        If not is_training: Returns preprocessed X DataFrame.
    """
    logger.info(f"Starting preprocessing. Mode: {'Training' if is_training else 'Inference'}")
    
    # 1. Select the columns that the model will actually use to learn.
    # We drop 'timestamp' because it's a date string. We already extracted recency features.
    
    # Check if 'impression' exists, if so drop it because it's always 1, giving no signal
    features_to_drop = ['timestamp', 'impression'] 
    
    # We keep user_id and offer_id right now just for grouping, but they shouldn't be used
    # directly as numerical features by the model (since they are arbitrary IDs).
    # We will exclude them from the actual training matrix later.
    
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns], errors='ignore')

    # 2. Categorical Encoding
    # What it does: Converts 'mobile_app' -> 0, 'web' -> 1, etc.
    # Why OrdinalEncoder instead of OneHotEncoder? Tree-based models (like XGBoost) handle ordinal 
    # encoded variables very well, and it keeps the dataframe size small.
    
    if is_training:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[config.CATEGORICAL_FEATURES] = encoder.fit_transform(X[config.CATEGORICAL_FEATURES])
        # Save the encoder so we use the EXACT same mapping during real-world inference
        save_object(encoder, config.ENCODER_SAVE_PATH)
    else:
        # Load the fitted encoder during API inference
        from src.utils import load_object # imported here to avoid circular imports if any
        encoder = load_object(config.ENCODER_SAVE_PATH)
        X[config.CATEGORICAL_FEATURES] = encoder.transform(X[config.CATEGORICAL_FEATURES])

    if not is_training:
        # For inference, just return the processed Features
        return X
        
    # 3. Train / Validation Split
    # What it does: Splits data so 80% is used for learning, 20% for testing (validation).
    # Why it's needed: If we test the model on the same data it learned from, it will just memorize 
    # the answers (Overfitting). We need to test on unseen data.
    
    y = X[config.TARGET_COL]
    # We drop the target column from X. We also drop ID columns because they hold no predictive math value.
    # We ONLY drop the IDs from X after extracting them if they are needed for grouping validation later.
    
    val_groups = X[config.GROUP_COL] # we need user_id to group by during Evaluation (MAP@7)
    
    X_model_features = X.drop(columns=[config.TARGET_COL, config.GROUP_COL, 'offer_id'])

    # Split: We use random split here.
    # In a perfect world, for logs, we use a Time-Based Split (train on past, test on future).
    # To keep this beginner-friendly, we use a standard random split.
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_model_features, y, val_groups, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
    )
    
    logger.info(f"Data split. Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # 4. Handle Severe Class Imbalance using SMOTE
    # What SMOTE does: Synthetic Minority Over-sampling Technique. It looks at the rare class (clicks)
    # and generates fake, realistic clicks by drawing lines between existing clicks in feature space.
    # Why it's needed: If only 2% of data are clicks, the ML model says "I'll just predict 0 every time 
    # and I'm 98% accurate!". SMOTE forces the model to learn what a click looks like.
    
    # Note on Architecture: SMOTE generates synthetic data rows. It is strictly for classification tasks.
    # Because we are using an XGBoost Classifier to simulate ranking (Point-wise Ranking),
    # applying SMOTE on the training set works perfectly to calibrate the classification scores!
    # IMPORTANT: Never apply SMOTE to Validation/Test data! We must validate on real-world distributions.
    
    logger.info("Applying SMOTE to training data to handle class imbalance...")
    smote = SMOTE(random_state=config.RANDOM_SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    logger.info(f"SMOTE complete. New train shape: {X_train_smote.shape}. "
                f"Clicks before: {sum(y_train)}, Clicks after: {sum(y_train_smote)}")

    # We return the groups for the validation set specifically so we can calculate MAP@7 per user later.
    return X_train_smote, X_val, y_train_smote, y_val, groups_val
