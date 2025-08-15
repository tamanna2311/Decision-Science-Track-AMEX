"""
File: ranking.py
Purpose: Define the model architecture used to rank offers.
Why it exists: Keeps the ML model definition decoupled from the training loop. If we want to swap 
               XGBoost for LightGBM later, we only change code here.
What it imports: xgboost.
Which files use it: train.py.
Inputs: Model hyperparameters.
Outputs: An untrained model object.
Role in the pipeline: It provides the 'engine' that will learn from the data.
"""

from xgboost import XGBClassifier
from src import config

def get_ranking_model(params=None):
    """
    Initializes a Point-wise Ranking Model.
    
    Why Classification for Ranking?
    True ranking algorithms (like true LambdaMART) optimize entire lists of items at once 
    (List-wise). However, a much simpler, widely-used, and highly effective industry approach 
    is "Point-wise Ranking".
    
    How it works:
    1. We train an XGBoost Classifier to predict the Probability of a Click (0.0 to 1.0).
    2. During inference, we score all candidate offers for a user.
    3. We sort the offers in descending order based on this probability score.
    4. The offer with the highest click probability becomes Rank 1.
    
    This is easier to understand, scales well, and natively supports class imbalance tools like SMOTE
    (which breaks down in true list-wise algorithms due to synthetic user groups).
    
    Inputs:
        params (dict): Dictionary of hyperparameters (useful during Optuna tuning).
        
    Outputs:
        XGBClassifier: Untrained model object.
    """
    # Base parameters that ensure reproducibility
    base_params = {
        'random_state': config.RANDOM_SEED,
        'n_estimators': 100, # We use early stopping, so this is just a max limit
        'objective': 'binary:logistic', # We want probabilities out
        'eval_metric': 'logloss'
    }
    
    # If custom parameters (from tuning) are provided, update the base parameters
    if params:
        base_params.update(params)
        
    # Return the initialized model
    model = XGBClassifier(**base_params)
    
    return model

def sort_by_score(candidate_offers_df, click_probabilities):
    """
    Given a list of offers and their predicted click probabilities, sorts them to create a ranking.
    Inputs: 
        candidate_offers_df: dataframe of offers
        click_probabilities: array of floats (from model.predict_proba)
    Outputs:
        Dataframe sorted by score.
    """
    candidate_offers_df = candidate_offers_df.copy()
    candidate_offers_df['score'] = click_probabilities
    
    # Sort descending, so highest probability is at the top
    ranked_df = candidate_offers_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    return ranked_df
