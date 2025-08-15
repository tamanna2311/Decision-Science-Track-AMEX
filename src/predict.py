"""
File: predict.py
Purpose: Define the inference logic (Prediction) used when the model runs live.
Why it exists: When the API receives a request, it doesn't need to tune or train. It just needs 
               to transform the incoming raw data exactly as we did during training, apply the 
               saved model, and sort the results.
What it imports: Everything needed for formatting features and loading models.
Which files use it: api/main.py.
Inputs: A user dictionary and a list of candidate offer dictionaries.
Outputs: A ranked list of offers with assigned probability scores.
Role in the pipeline: The bridge between the trained static model and the real-time API.
"""

import pandas as pd
from src.feature_engineering import generate_features
from src.preprocess import preprocess_data
from src.ranking import sort_by_score
from src import config
from src.utils import load_object, get_logger

logger = get_logger(__name__)

# Global variables to hold loaded models so we don't load them on EVERY single API request
# (which would be very slow). They will be initialized once when the API starts.
_RANKER_MODEL = None

def load_artifacts():
    """
    Loads saved model artifacts into memory.
    Why it exists: Disk I/O is slow. Loading models takes time. By loading them once 
    into global variables, the API can serve thousands of requests instantly.
    """
    global _RANKER_MODEL
    if _RANKER_MODEL is None:
        logger.info("Loading model artifact from disk...")
        _RANKER_MODEL = load_object(config.MODEL_SAVE_PATH)

def rank_candidate_offers(user_data: dict, candidate_offers: list) -> list:
    """
    Takes a massive raw payload of user and offer context, prepares the features, 
    runs prediction, and outputs a ranked list.
    
    Why this exists:
    The frontend (like a mobile app) sends raw context. We must simulate exactly what the `train.py`
    script did to the data before feeding it to the model.
    
    Inputs:
        user_data (dict): Demographic and behavioural info about the user making the request.
        candidate_offers (list): A list of dictionaries, representing available offers to show them.
        
    Outputs:
        list of dicts containing the offer ID and its assigned probability.
    """
    # 1. Combine user data and candidate offers into a single DataFrame
    # Why? We need to construct rows where each row represents User + Offer combo.
    expanded_rows = []
    for offer in candidate_offers:
        row = user_data.copy()
        row.update(offer)
        expanded_rows.append(row)
        
    df = pd.DataFrame(expanded_rows)
    
    # 2. Feature Engineering
    # We apply the same transformation logic (CTRs, Freuqency, Popularity computations).
    # Since we are doing this live, we would normally fetch historical CTRs from a fast database (Redis).
    # For this simplified project, we pretend the API request sends along enough context 
    # to calculate these features directly using our engineering function.
    df_features = generate_features(df)
    
    # 3. Preprocessing
    # Transform strings to encoded numbers using the encoder we SAVED during training.
    # is_training=False ensures we don't accidentally do SMOTE or train/test splits.
    X_inference = preprocess_data(df_features, is_training=False)
    
    # We drop the IDs since the model doesn't accept them
    X_model_features = X_inference.drop(columns=[config.GROUP_COL, 'offer_id'], errors='ignore')
    
    # 4. Predict
    # We use our globally loaded model context to score the rows
    load_artifacts()
    # predict_proba returns an array like [[0.9, 0.1], [0.3, 0.7]]. We want the second column (Index 1)
    scores = _RANKER_MODEL.predict_proba(X_model_features)[:, 1]
    
    # 5. Rank
    # We attach the scores to the original df and sort them highest to lowest.
    ranked_df = sort_by_score(df, scores)
    
    # 6. Format Response
    # The API output needs to be clean JSON, so we just return the ID and Score.
    response = []
    # Using iterrows for simplicity, though vectorization is better for massive matrices
    for idx, row in ranked_df.iterrows():
        response.append({
            "offer_id": int(row['offer_id']),
            "score": float(row['score']), # Convert numpy float to native float for JSON
            "rank": idx + 1
        })
        
    return response

