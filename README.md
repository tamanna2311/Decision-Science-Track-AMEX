# Personalization of Amex Digital Offers

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Ranking-green.svg)
![Optuna](https://img.shields.io/badge/Optuna-Tuning-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-teal.svg)

## 1. Project Overview
This project is an end-to-end Decision Science pipeline for ranking digital offers. 
Given a specific user and a set of available candidate offers, the system predicts the probability of the user clicking on each offer and returns a ranked list (most relevant on top). 
It features synthetic data generation, advanced feature engineering, hyperparameter tuning, and a real-time REST API.

## 2. Problem Statement
Users log into their digital banking apps and see "Offers" (e.g., "$5 off coffee", "10% back on travel"). However, screen space is limited. If we show a coffee offer to a user who doesn't drink coffee, we waste an impression and lose potential revenue. The goal is to maximize the Click-Through Rate (CTR) by showing the *right* offer to the *right* person.

## 3. Why Personalization Matters
- **For the Business:** Higher clicks lead to higher transaction volumes and merchant partnerships.
- **For the User:** Less clutter. They only see deals they actually care about, improving trust in the brand.

## 4. Why Ranking Instead of Just Classification?
Classification asks: *"Will the user click this offer? (Yes/No)"*  
Ranking asks: *"Out of these 10 offers, what is the exact order I should show them?"*
  
If our classification model predicts "Yes" for 5 out of 10 offers, it doesn't tell us which of those 5 to put at the very top. Ranking solves this by giving every offer a continuous score, optimizing the entire *list* presented to the user.

## 5. MAP@7 Explained Simply
**Mean Average Precision at K (MAP@K)** is the gold standard for ranking.
Imagine you have 7 slots on the app screen. 
- If the user's favorite offer is placed at slot #1, you get maximum points!
- If it's placed at slot #7, you get fewer points. 
- If it's placed at slot #8 (off-screen), you get 0 points.
MAP@7 calculates the sum of precision at every rank where a correct click happens, giving a massive bias towards putting the absolute best offers at the very top.

## 6. LambdaMART & Point-wise Ranking Explained Simply
LambdaMART is an algorithm designed to optimize List-wise ranking directly via Gradient Boosting. However, true List-wise optimization is extremely complex to set up and balance against severe class imbalance (e.g., only 2% of offers get clicked). 
**Our Solution (Point-wise Ranking with XGBoost):** We train a powerful XGBoost Classifier on balanced data (using SMOTE), extract the highly calibrated probabilities (`predict_proba`), and sort the offers descending by that probability. This achieves the business goal of ranking while maintaining a clean, understandable, and scalable data pipeline.

## 7. Folder Structure
```text
AmexPersonalization/
├── data/                  # Contains raw synthetic logs and processed ML data
├── models/                # Saved XGBoost models and Sklearn Encoders
├── src/
│   ├── config.py          # Master configuration (paths, hyperparams)
│   ├── data_generation.py # Simulates massive interaction logs safely
│   ├── feature_engineering.py # Creates historical CTRs and frequency stats
│   ├── preprocess.py      # Cleans, Encodes, SMOTEs, and Splits data
│   ├── ranking.py         # The Point-wise XGBoost Ranking logic
│   ├── evaluate.py        # Custom MAP@7 metric calculation
│   ├── train.py           # The Optuna Tuning and Training loop
│   ├── predict.py         # Transforms real-time API payloads to ML formats
│   └── utils.py           # Logging and artifact loading
├── api/
│   ├── main.py            # FastAPI server
│   └── schemas.py         # Pydantic data validation
├── tests/
│   └── test_api.py        # Automated Pytest suite
├── run_pipeline.py        # One-click execution script
└── architecture_explained.md # Deep dive into the flow!
```

## 8. Step-by-Step Run Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the full ML Pipeline (Data Generation -> Training -> Evaluation):**
   ```bash
   python run_pipeline.py
   ```
   *Watch the terminal! We log every single step explicitly so you can see SMOTE working, Optuna tuning, and the MAP@7 score.*
3. **Start the API Server:**
   ```bash
   uvicorn api.main:app --reload
   ```
4. **Run the Automated Tests (Optional):**
   ```bash
   pytest tests/
   ```

## 9. Example API Request & Response
With the server running at `http://127.0.0.1:8000`, send this POST to `/rank_offers`:

**Request (JSON):**
```json
{
  "user": {
    "user_id": 105,
    "device_type": "mobile_app",
    "user_segment": "high_spender",
    "previous_clicks": 3,
    "previous_impressions": 40,
    "time_since_last_activity": 1.5,
    "transaction_count": 8
  },
  "candidate_offers": [
    {"offer_id": 12, "offer_category": "dining"},
    {"offer_id": 45, "offer_category": "travel"},
    {"offer_id": 89, "offer_category": "retail"}
  ]
}
```

**Output:** The API returns them ranked. Offer 45 is the winner!
```json
{
  "ranked_offers": [
    {"offer_id": 45, "score": 0.892, "rank": 1},
    {"offer_id": 12, "score": 0.310, "rank": 2},
    {"offer_id": 89, "score": 0.124, "rank": 3}
  ]
}
```

## 10. Explanation of Generated Features
Our ML model does not read raw data. We transform the data into:
- **Historical CTR:** Baseline popularity. Does this user click often? Is this offer broadly loved?
- **Activity Frequency:** High frequency implies high intent right now.
- **Offer Popularity:** Tells the model whether a high CTR is a fluke on a small sample or a verified trend.

## 11. Explanation of Optuna Tuning and Cross-Validation
We use **Optuna** to try combinations of learning rates and tree depths.
Because data science shouldn't rely on luck! Optuna uses Bayesian optimization to intelligently home in on the best settings quickly. We validate on an isolated hold-out set to ensure the model isn't just "memorizing" the training data.

## 12. Interview Explanation Section

**How to pitch this in an interview:**
"I built an end-to-end Offer Ranking engine. Because I didn't have access to proprietary Amex data, I wrote a robust data generator to simulate 60,000 interaction logs mapping users to offers, injecting realistic class imbalance. I engineered temporal and CTR features, handled the severe lack of clicks using SMOTE, and built a Point-wise ranking system using XGBoost. To prove it worked in the real world, I skipped standard accuracy metrics and evaluated the model strictly using MAP@7. Finally, I wrapped the inference engine in a FastAPI microservice."

**Likely Interview Questions:**
1. **Why XGBoost over Deep Learning?** Tabular data (clicks, categories, counts) performs incredibly well on Tree-based models. XGBoost is faster to train, easier to tune, and highly explainable compared to Neural Networks.
2. **What happens if a new user joins (Cold Start Problem)?** Since they have 0 historical clicks, the model relies heavily on the `Offer Popularity` and global `Offer CTR` features, gracefully defaulting to showing them universally trending offers until they build history.
3. **Why did you use SMOTE?** Clicks are rare (maybe 2% CTR). Without SMOTE oversampling the minority class, the model learns a lazy strategy: "Just predict 0 every time." SMOTE forces it to map the decision boundaries of a 'click'.
4. **Why did you choose Point-wise ranking instead of Pairwise?** Pairwise models (like strict LambdaMART) are powerful but complex to integrate with synthetic oversampling (SMOTE breaks apart pair structures). Point-wise is an extremely common, highly scalable industry standard where we predict probabilities and sort them later.
