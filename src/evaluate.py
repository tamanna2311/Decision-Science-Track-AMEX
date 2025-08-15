"""
File: evaluate.py
Purpose: Calculate specific evaluation metrics for ranking, primarily MAP@7.
Why it exists: Standard classification metrics (Accuracy, F1) are terrible for ranking. 
               If I recommend 100 things and only 1 is relevant, F1 penalizes me heavily. 
               But if that 1 relevant thing is the #1 item on the list, a ranking system did a PERFECT job.
What it imports: numpy, pandas.
Which files use it: train.py.
Inputs: Lists of true clicks, Lists of predicted probabilities, User Groups.
Outputs: Numerical score (MAP@7).
Role in the pipeline: Judges how well the trained model actually performs business-wise.
"""

import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

def average_precision_at_k(y_true, y_scores, k=7):
    """
    Calculates Average Precision at K (AP@K) for a single user's list of offers.
    
    What is AP@K in Plain English?
    Imagine we recommend 7 offers to a user. We only care about the top 7.
    If the user clicks on the 1st offer, that's a huge win!
    If they click on the 7th offer, that's okay, but not as good as the 1st.
    AP@K calculates a score that gives MORE points for getting clicks at the very top of the list, 
    and fewer points for clicks at the bottom.
    
    Formula:
    AP@K = Sum of (Precision at rank i * change in Recall at rank i)
    This effectively averages the precision at every point we find a relevant (clicked) item.
    
    Inputs:
        y_true (list/array): The actual clicks (e.g., [1, 0, 0, 1])
        y_scores (list/array): The predicted probabilities (e.g., [0.9, 0.4, 0.2, 0.1])
        k (int): How deep down the list we care about (default 7).
        
    Outputs:
        float: The AP score for this list.
    """
    # 1. Combine the true labels and predicted scores
    items = list(zip(y_true, y_scores))
    
    # 2. Sort the list based on the predicted scores (highest score first)
    # This simulates how the Recommendation Engine visually shows the list to the user.
    items.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Keep only the top K items
    items = items[:k]
    
    # 4. Calculate Precision at each step
    score = 0.0
    num_hits = 0.0 # How many clicks we've seen so far in the top K
    
    for i, (actual_click, _) in enumerate(items):
        if actual_click == 1:
            num_hits += 1.0
            # precision at this specific rank 'i' (ranks are 1-indexed, so i+1)
            precision_at_this_rank = num_hits / (i + 1.0) 
            score += precision_at_this_rank
            
    # 5. Divide by the total possible hits.
    # Wait, what if the user actually clicked on 0 things overall? 
    # AP is 0 to avoid dividing by zero.
    total_relevant = sum(y_true)
    if total_relevant == 0:
        return 0.0
        
    # Standard definition: divide by the minimum of K or total actual clicks
    return score / min(len(items), total_relevant)

def map_at_k(y_true, y_scores, groups, k=7):
    """
    Calculates Mean Average Precision at K (MAP@K) across ALL users.
    
    What is MAP@K?
    We calculate AP@K for User 1, AP@K for User 2, AP@K for User 3...
    Then we just take the mean (average) of all those scores.
    This gives us one single number to evaluate our entire system.
    
    Why use groups?
    Ranking is inherently a grouped problem. We don't rank all 100,000 offers against each other.
    We rank User 1's 10 offers against each other. Therefore, we must calculate the metric *per user group*.
    
    Inputs:
        y_true: True clicks (pandas series)
        y_scores: Predicted probabilities (numpy array)
        groups: The user IDs corresponding to each row (pandas series)
        k: The cutoff rank.
        
    Outputs:
        float: The final MAP score (higher is better, max 1.0)
    """
    df = pd.DataFrame({
        'group': groups,
        'y_true': y_true,
        'y_score': y_scores
    })
    
    ap_scores = []
    
    # Iterate through each user's unique list of offers
    for _, group_data in df.groupby('group'):
        ap = average_precision_at_k(group_data['y_true'].values, group_data['y_score'].values, k)
        ap_scores.append(ap)
        
    # Return the mean of all AP scores
    mean_ap = np.mean(ap_scores)
    
    return mean_ap

# --- WORKED EXAMPLE FOR COMMENTS ONLY ---
# Imagine a user is shown 4 offers. 
# True Clicks: [0, 1, 0, 1]
# Model Scores: [0.3, 0.8, 0.1, 0.9]
# 1. Sort by scores descending:
#    Score 0.9 -> True Click 1 (Rank 1)
#    Score 0.8 -> True Click 1 (Rank 2)
#    Score 0.3 -> True Click 0 (Rank 3)
#    Score 0.1 -> True Click 0 (Rank 4)
# 2. Calculate Precision at each rank where a hit occurs:
#    Rank 1 is a hit: Precision = 1/1 = 1.0
#    Rank 2 is a hit: Precision = 2/2 = 1.0
# 3. Sum these precisions: 1.0 + 1.0 = 2.0
# 4. Divide by total hits (which is 2): AP = 2.0 / 2 = 1.0
# This makes sense! Our model put the two correct offers at the very top. Perfect AP score!
