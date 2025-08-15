"""
File: data_generation.py
Purpose: Generate realistic synthetic logs of user-offer interactions.
Why it exists: Real company data (like Amex) is highly confidential and massive. We need a way 
               to simulate it so we can build and test our ML pipeline locally.
What it imports: pandas to create dataframes, numpy for random sampling, and config for settings.
Which files use it: run_pipeline.py calls this first to create the starting dataset.
Inputs: None (it uses settings from config.py).
Outputs: Saves a CSV file to 'data/raw/synthetic_logs.csv' and returns a pandas DataFrame.
Role in the pipeline: Step 1. It provides the foundational raw data simulating the real world.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src import config
from src.utils import get_logger

# Initialize logger
logger = get_logger(__name__)

def generate_synthetic_data() -> pd.DataFrame:
    """
    Simulates user interactions with digital offers over time.
    
    Why this exists:
    We need complex, messy data that resembles real behavior. For example, some users are highly 
    active, some offers are inherently more popular, and clicks depend partially on the user/offer context.
    
    Logic & Simplifications:
    1. We generate user IDs and assign them a random segment.
    2. We generate offer IDs and assign them categories.
    3. We simulate random events (impressions).
    4. We generate 'clicks' with a severe class imbalance (most impressions don't result in clicks),
       which simulates the real-world challenge of CTR (Click-Through Rate) prediction.
    
    Outputs:
        pd.DataFrame containing the generated simulated data.
    """
    logger.info("Starting synthetic data generation...")
    
    # 1. Set the random seed so that every time we run this, we get the exact same "random" data.
    # This is crucial for debugging and reproducibility.
    np.random.seed(config.RANDOM_SEED)
    
    # 2. Define lists of possible categorical values
    device_types = ['mobile_app', 'web', 'tablet']
    offer_categories = ['dining', 'travel', 'retail', 'entertainment', 'grocery']
    user_segments = ['high_spender', 'frequent_traveler', 'cashback_seeker', 'new_user', 'dormant']
    
    # 3. Create arrays of randomized data for our logs
    # We sample randomly from our defined lists or distributions to build rows of data.
    logger.info(f"Generating {config.NUM_LOGS} rows for {config.NUM_USERS} users and {config.NUM_OFFERS} offers.")
    
    user_ids = np.random.randint(1, config.NUM_USERS + 1, config.NUM_LOGS)
    offer_ids = np.random.randint(1, config.NUM_OFFERS + 1, config.NUM_LOGS)
    
    # We want timestamps spread over the last 30 days
    now = datetime.now()
    timestamps = [now - timedelta(days=np.random.uniform(0, 30)) for _ in range(config.NUM_LOGS)]
    
    # Randomly assign categorical attributes
    devices = np.random.choice(device_types, config.NUM_LOGS, p=[0.6, 0.3, 0.1]) # 60% mobile
    categories = np.random.choice(offer_categories, config.NUM_LOGS)
    segments = np.random.choice(user_segments, config.NUM_LOGS)

    # Generate historical behavioural stats with realistic distributions
    # e.g., using exponential distribution to simulate that most people have few previous clicks,
    # and very few have high previous clicks.
    prev_clicks = np.random.geometric(p=0.5, size=config.NUM_LOGS) - 1
    prev_impressions = prev_clicks + np.random.randint(1, 20, config.NUM_LOGS)
    
    # Time since last activity in hours
    time_since_last = np.random.exponential(scale=48, size=config.NUM_LOGS) 
    
    # Number of recent transactions
    transactions = np.random.poisson(lam=2, size=config.NUM_LOGS)

    # 4. Build a base probability for a click
    # Realism check: We deliberately create a signal. 
    # For instance, if user is high_spender, probability goes up.
    # If device is mobile, probability might change.
    # But overall CTR is usually very low (like 1-5%).
    click_probs = np.random.uniform(0.01, 0.05, config.NUM_LOGS)
    
    # Let's add slight biases so the model has something to learn
    # Increase prob for dining offers shown to frequent travelers (just an arbitrary synthetic rule)
    dining_traveler_mask = (categories == 'dining') & (segments == 'frequent_traveler')
    click_probs[dining_traveler_mask] += 0.05 
    
    # 5. Generate the target variable 'click' based on the probabilities
    # 1 for click, 0 for no click (impression only).
    clicks = np.random.binomial(n=1, p=click_probs)
    
    logger.info(f"Generated clicks with a baseline anomaly rate. Total clicks: {sum(clicks)} / {config.NUM_LOGS}")

    # 6. Put everything into a pandas DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'offer_id': offer_ids,
        'timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamps],
        'device_type': devices,
        'offer_category': categories,
        'user_segment': segments,
        'previous_clicks': prev_clicks,
        'previous_impressions': prev_impressions,
        'time_since_last_activity': time_since_last,
        'transaction_count': transactions,
        'impression': np.ones(config.NUM_LOGS, dtype=int), # Every log is an impression
        'click': clicks # The target variable
    })
    
    # Sort by time to make it realistic (logs act as time-series)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # 7. Save the raw data
    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)
    
    logger.info(f"Synthetic data generation complete. Saved to {config.RAW_DATA_PATH}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
