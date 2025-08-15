"""
File: train.py
Purpose: Define the training loop, including Hyperparameter Tuning via Optuna.
Why it exists: This is where the model actually "learns". We connect the data to the model,
               tune the parameters to find the best configuration, evaluate it, and save it.
What it imports: Everything! Preprocessing, ranking model, evaluation metrics, saving utilities.
Which files use it: run_pipeline.py calls it to execute training.
Inputs: Raw pandas dataframe.
Outputs: Saves the best model to disk.
Role in the pipeline: The core learning phase.
"""

import optuna
import pandas as pd
from typing import Tuple

from src.preprocess import preprocess_data
from src.ranking import get_ranking_model
from src.evaluate import map_at_k
from src.utils import get_logger, save_object
from src import config

logger = get_logger(__name__)

def train_and_evaluate(df: pd.DataFrame):
    """
    Runs the complete training process including feature extraction, tuning, and evaluation.
    
    Why Optuna?
    Machine Learning algorithms have settings called 'Hyperparameters' (like how deep a decision tree goes, 
    or how fast the model learns). We don't know the best settings in advance. 
    Optuna is an automated framework that tries out combinations intelligently, rather than us guessing.
    We are using a small number of trials (3) to keep it fast, but in a real project it would run 100+ times.
    """
    logger.info("Starting training pipeline...")

    # 1. Preprocess the data
    # We get matrices ready for XGBoost. 
    X_train, X_val, y_train, y_val, val_groups = preprocess_data(df, is_training=True)
    
    # 2. Define the Optuna Objective Function
    # What it does: This defines a single experiment (a 'trial'). Optuna will call this function 
    # multiple times, passing in different parameter values. Our mission is to return a score 
    # for Optuna to maximize (MAP@7).
    def objective(trial):
        # Suggest parameters to test
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        
        # Instantiate the model with these suggested parameters
        model = get_ranking_model(params)
        
        # Train the model
        # Note: We don't use early stopping inside the Optuna objective to keep it clean,
        # but we could use evaluating on the validation set.
        model.fit(X_train, y_train)
        
        # Predict probability of click on the validation set
        y_val_preds = model.predict_proba(X_val)[:, 1] # Index 1 is the probability of class 1 (Click)
        
        # Calculate our custom business-centric metric: MAP@7
        score = map_at_k(y_val, y_val_preds, val_groups, k=config.K_FOR_MAP)
        
        return score

    # 3. Create the Optuna Study
    # We tell Optuna we want to 'maximize' the score, since a higher MAP is better.
    logger.info("Beginning Hyperparameter Tuning with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=config.OPTUNA_TRIALS)
    
    logger.info(f"Best Optuna Trial: {study.best_trial.value}")
    logger.info(f"Best Parameters: {study.best_params}")

    # 4. Train Final Model on Best Parameters
    # We found the best settings! Now we fit a final model using them.
    logger.info("Training final model using best parameters...")
    final_model = get_ranking_model(study.best_params)
    
    # Here we utilize early stopping. If the validation metric (logloss) stops improving
    # for 10 rounds, training halts to prevent overfitting.
    # *Note on fit param format: XGBoost >= 1.6 changed how early stopping is handled, 
    # using eval_set during fit is standard.
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
        verbose=False
    )
    
    # 5. Final Evaluation
    y_val_final_preds = final_model.predict_proba(X_val)[:, 1]
    final_map_score = map_at_k(y_val, y_val_final_preds, val_groups, k=config.K_FOR_MAP)
    
    logger.info(f"Final Validation MAP@{config.K_FOR_MAP}: {final_map_score:.4f}")
    
    # 6. Save the Model
    # We must save the model into an artifact (.joblib) so our FastAPI API can load it 
    # without having to re-run this entire script.
    save_object(final_model, config.MODEL_SAVE_PATH)
    
    return final_model
