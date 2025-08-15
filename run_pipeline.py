"""
File: run_pipeline.py
Purpose: Connect all the 'src' components together to execute the entire ML lifecycle in one command.
Why it exists: If a colleague joins the project, they shouldn't have to manually run 5 different python 
               scripts in a magical order. This script orchestrates Data Gen -> Features -> Train -> Eval.
What it imports: Top-level functions from the src/ folder modules.
Which files use it: The human user runs this from the terminal.
Inputs: None.
Outputs: Trained models saved to disk.
Role in the pipeline: The Maestro/Orchestrator.
"""

from src.data_generation import generate_synthetic_data
from src.feature_engineering import generate_features
from src.train import train_and_evaluate
from src.utils import get_logger

logger = get_logger(__name__)

def main():
    """
    Executes the full pipeline.
    """
    logger.info("========================================")
    logger.info("  AMEX OFFER PERSONALIZATION PIPELINE   ")
    logger.info("========================================")
    
    # Step 1: Generate Data
    logger.info("\n--- STEP 1: Creating Synthetic Data ---")
    raw_df = generate_synthetic_data()
    
    # Step 2: Feature Engineering
    logger.info("\n--- STEP 2: Engineering Features ---")
    featured_df = generate_features(raw_df)
    
    # Step 3: Model Training, Tuning, and Evaluation
    logger.info("\n--- STEP 3: Training & Evaluating Model ---")
    train_and_evaluate(featured_df)
    
    logger.info("\n========================================")
    logger.info("  PIPELINE COMPLETE - ARTIFACTS SAVED   ")
    logger.info("========================================")

if __name__ == "__main__":
    main()
