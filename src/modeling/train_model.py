import logging
import pickle
from pathlib import Path
from typing import Dict, Any
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

def retrain_model(model: BaseEstimator, data: pd.DataFrame, config: Dict[str, Any]) -> BaseEstimator:
    """
    Retrain the model using the new data.
    
    Args:
        model: Pre-trained model to be retrained.
        data: DataFrame containing the new features and labels.
        config: Dictionary containing the model configuration, including the target column name.
        
    Returns:
        Retrained model.
    """
    try:
        y_full = data[config['target_col']]
        #print shape of y-full
        x_full = data[config['feature_col']]

        vectorizer= TfidfVectorizer()
        tf_x_full = vectorizer.fit_transform(x_full)

        # Fit the model with the new data
        model.fit(tf_x_full, y_full)
    except KeyError as e:
        logger.error(f"KeyError: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during model retraining: {e}")
        raise

    return vectorizer, model

def save_model(model: BaseEstimator, path: Path) -> None:
    """
    Save the model to disk.
    
    Args:
        model: The model to be saved.
        path: The path where the model should be saved.
    """
    try:
        with path.open('wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        logger.error(f"An error occurred during model saving: {e}")
        raise

def save_vectorizer(vectorizer, path):
    """
    Save the vectorizer to disk.
    
    Args:
        vectorizer: The vectorizer to be saved.
        path: The path where the vectorizer should be saved.
    """
    try:
        with path.open('wb') as f:
            pickle.dump(vectorizer, f)
    except Exception as e:
        logger.error(f"An error occurred during model saving: {e}")
        raise
