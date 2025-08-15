"""
File: utils.py
Purpose: Provide reusable utility functions (logging setup, saving/loading files) so we don't repeat code.
Why it exists: It follows the "DRY" (Don't Repeat Yourself) principle. Every script needs logging and I/O.
What it imports: logging (to print messages), joblib (to save/load Python objects), and pathlib.
Which files use it: Almost every script in 'src/' and 'api/'.
Inputs: Varied, depending on the function.
Outputs: Varied, logging objects or loaded models.
Role in the pipeline: A helper module providing shared functionality.
"""

import logging
import joblib
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a customized logger.
    
    Why this exists:
    Instead of using standard `print()` statements, logging allows us to add timestamps, 
    severity levels (INFO, WARNING, ERROR), and easily redirect output to files later if needed.
    
    Inputs:
        name (str): The name of the module requesting the logger (usually __name__).
        
    Outputs:
        logging.Logger: A configured logger object.
    """
    # Create a logger object
    logger = logging.getLogger(name)
    
    # Set the threshold for logging. INFO means it will show INFO, WARNING, ERROR, CRITICAL.
    logger.setLevel(logging.INFO)
    
    # Check if the logger already has handlers to prevent duplicate messages when imported multiple times
    if not logger.handlers:
        # Create a console handler (prints to terminal)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Define how the log messages will look: "[Timestamp] - Name - LEVEL - Message"
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Attach the handler to the logger
        logger.addHandler(ch)
        
    return logger

def save_object(obj, filepath: Path):
    """
    Saves a Python object (like a trained model or scaler) to a file.
    
    Why this exists:
    We need to persist our models to disk so the API can load them later without retraining.
    We use joblib because it is highly optimized for saving scikit-learn/numpy arrays.
    
    Inputs:
        obj: The Python object to save.
        filepath (Path): The destination file path.
    """
    # Ensure the parent directory exists before saving
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the object using joblib
    joblib.dump(obj, filepath)
    logger = get_logger(__name__)
    logger.info(f"Successfully saved object to {filepath}")

def load_object(filepath: Path):
    """
    Loads a Python object from a file.
    
    Why this exists:
    When predicting/inference happens, we must load the exact same model & encoders 
    used during training.
    
    Inputs:
        filepath (Path): The file path to load from.
        
    Outputs:
        The loaded Python object.
    """
    logger = get_logger(__name__)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
        
    # Load and return the object
    obj = joblib.load(filepath)
    logger.info(f"Successfully loaded object from {filepath}")
    return obj
