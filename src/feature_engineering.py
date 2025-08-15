"""
File: feature_engineering.py
Purpose: Transform raw transactional/log data into meaningful 'features' the ML model can understand.
Why it exists: A machine learning model can't easily interpret a raw list of interactions. It needs 
               summarized numbers expressing patterns, like "How frequently does this user click?" 
               or "How popular is this offer?". Feature engineering is the most critical step in Data Science.
What it imports: pandas.
Which files use it: preprocess.py and the API (for real-time feature generation).
Inputs: Raw pandas DataFrame.
Outputs: Enriched pandas DataFrame with new columns.
Role in the pipeline: Step 2. Converts base data into rich, predictive signals.
"""

import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies multiple transformations to generate predictive features.
    
    Why this exists:
    By creating aggregate statistics (like CTR), we provide the model with "historical memory" 
    so it knows whether a user likes a specific feature without memorizing every past log.
    
    Inputs:
        df (pd.DataFrame): Raw interaction logs.
        
    Outputs:
        pd.DataFrame: Enriched dataframe containing historical CTR, frequency, and popularity.
        
    Simplification note:
    In a real system, we must ensure "Point-in-Time" correctness to avoid data leakage 
    (i.e., not using the *future* to predict the *past*). To keep this beginner-friendly 
    and simple, we compute global aggregates. In a strict real-world pipeline, we'd use 
    expanding window functions or lagging.
    """
    logger.info("Starting feature engineering...")
    
    # Copy DataFrame to avoid modifying the original implicitly
    df = df.copy()

    # ==========================================
    # Feature 1 & 2: Historical CTRs (Click-Through Rates)
    # ==========================================
    # What it does: Calculates (Total Clicks / Total Impressions) for each user and each offer.
    # Why it's needed: CTR is the strongest signal for relevance. If a user usually clicks, 
    # their baseline probability is higher. If an offer is highly clicked, it's globally "popular".
    
    logger.info("Calculating historical CTR features...")
    
    # Calculate User historical CTR
    # We group by user, take the mean of 'click' (since click is 1 or 0, mean = CTR).
    # We use a trick with transform('mean') to broadcast the result back to every row belonging to the user.
    df['user_hist_ctr'] = df.groupby('user_id')['click'].transform('mean')
    
    # Calculate Offer historical CTR
    df['offer_hist_ctr'] = df.groupby('offer_id')['click'].transform('mean')
    
    # ==========================================
    # Feature 3: Activity Frequency
    # ==========================================
    # What it does: Approximates how frequently an interaction occurs.
    # Why it's needed: Engaged users behave differently than dormant users. 
    # High frequency might indicate high intent.
    # Logic: We divide the transaction_count by the time_since_last_activity.
    # We add 1 to the denominator to avoid DivisionByZero errors.
    
    logger.info("Calculating activity frequency features...")
    df['activity_frequency'] = df['transaction_count'] / (df['time_since_last_activity'] + 1)
    
    # ==========================================
    # Feature 4: Offer Popularity (Raw Count)
    # ==========================================
    # What it does: Counts total number of times an offer was shown.
    # Why it's needed: An offer might have a high CTR just because it was shown rarely to a perfect audience.
    # Giving the model the raw volume helps it trust the CTR metric more or less.
    df['offer_popularity'] = df.groupby('offer_id')['impression'].transform('sum')

    # Note on Temporal features: We could extract hour of day from timestamp, etc., 
    # which is common, but we'll stick to core behavioural features here to maintain simplicity.

    logger.info(f"Feature engineering complete. Added new columns. Total columns: {df.shape[1]}")
    
    return df
