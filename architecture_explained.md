# Architecture Explained
_A deep dive into how data moves through this project, from absolute scratch to a live API output._

This document is designed to help you completely understand the "Why" and "How" of the system architecture.

---

## The Complete Flow (Bird's Eye View)

1. **`data_generation.py`** -> Creates the universe.
2. **`feature_engineering.py`** -> Gives the universe meaning (CTR, Trends).
3. **`preprocess.py`** -> Translates the meaning into pure math (Encoding, SMOTE).
4. **`ranking.py` & `evaluate.py`** -> The brain that learns the math and scores itself.
5. **`train.py`** -> The gym where the brain works out via Optuna.
6. **`predict.py`** -> The translator that takes raw web data and feeds it to the brain.
7. **`main.py`** -> The web window that lets humans talk to the translator.

---

## 1. Raw Synthetic Data Generation
We start with nothing. `data_generation.py` simulates reality. It essentially iterates through tens of thousands of rows asking: "Did Mobile User 104 click on the Dining Offer today?" Most importantly, it creates **Severe Class Imbalance**. In reality, 98% of people scroll past ads. Generating a dataset where everyone clicks is unrealistic. We inject a 2-5% baseline click rate.

## 2. The Power of Feature Engineering
A machine learning algorithm looks at a row: `User: 104, Offer: Dining`. It has no memory. It doesn't inherently go "Ah, User 104 loves dining!"
`feature_engineering.py` computes aggregates over the entire dataset. It creates a new column called `user_hist_ctr`. If User 104 clicked on 8 out of 10 dining offers in the past, that column now holds `0.80`. Suddenly, the model is given a golden ticket. It learns: "If `user_hist_ctr` is high, output a high probability."

## 3. The Math Translation (Preprocessing)
Models don't speak English. 
`preprocess.py` performs **Ordinal Encoding**. It takes `['mobile_app', 'web']` and secretly maps them to `[0, 1]`. Crucially, it **saves that secret map (the encoder)** to the hard drive (`models/ordinal_encoder.joblib`). If we forget the map, when a user asks the API for a prediction on a `mobile_app` exactly five days later, the model will crash.

It also introduces **SMOTE**. Let's say we have 10,000 rows, and only 200 are clicks. SMOTE looks at those 200 clicks in a mathematical space, draws lines between them, and invents *new*, realistic synthetic clicks until we have 9,800 clicks and 9,800 non-clicks. Now the model has enough examples to actually learn what makes a click special.

## 4. The Model Training Workout Loop
`train.py` leverages **Optuna**. Optuna is an auto-tuning tool. Instead of us manually guessing "Should the decision tree be 3 levels deep or 6 levels deep?", Optuna sets up an experiment. 
It says: "I'll try 3! (Trains model... evaluates... records score of 0.50 MAP)."
"Now I'll try 6! (Trains model... evaluates... records score of 0.61 MAP)."
It finds the best combination and locks it in. Once complete, it saves the brain to `models/ranker_model.joblib`.

## 5. Judging the Model (MAP@7)
`evaluate.py` handles the most complex business logic. Let's say User 5 has 10 possible offers assigned to them during validation. 
Our model looks at them and assigns probability scores: [0.99, 0.20, 0.50...].
We sort those score descending. 
If the offer the user *actually* clicked happens to be at the top of that sorted list, MAP@7 gives us a nearly perfect 1.0 score. If it's at the bottom, we get practically a 0.0. 
We average this score across all thousands of users, giving us our final metric.

## 6. Going Live (The API Inference)
The training is over. A customer opens the Amex App. The Amex App sends an HTTP Payload to `main.py` (FastAPI).

The payload contains the customer's attributes, and 5 candidate offers Amex wants to rank.
`predict.py` takes over.
1. It combines the customer + 5 offers into a 5-row Dataframe.
2. It calls `feature_engineering.py` exactly as training did.
3. It loads the `ordinal_encoder` from disk, translating 'mobile_app' to 0.
4. It loads the `ranker_model` from disk.
5. It feeds the 5 perfectly formatted math rows to the model.
6. The model returns 5 probabilities: `[0.1, 0.8, 0.4, 0.02, 0.9]`
7. `predict.py` binds these scores to the Offer IDs and sorts them highest to lowest.
8. `main.py` sends the sorted JSON back to the App.

The highest probability offer is rendered at the absolute top of the user's phone screen.
The pipeline is complete.
