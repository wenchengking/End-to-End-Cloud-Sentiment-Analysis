import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

def score_model(test: pd.DataFrame, model: LogisticRegression, config: dict) -> pd.DataFrame:
    """
    Score the trained model on the test dataset using the specified configuration.

    Args:
        data: Tuple containing the test features and labels.
        model: Trained LogisticRegression model.
        config: Dictionary containing the configuration for scoring.

    Returns:
        DataFrame containing the test dataset with predicted probability and binary outputs.
    """
    try:
        logger.debug("Preparing to score the model with configuration: %s", config)
        
        #split the data into fetures and labels
        x_test = test.drop(columns=[config["get_target"]["target"]])
        y_test = test[config["get_target"]["target"]]

        # Get predicted probabilities and binary outputs for the test set
        ypred_proba_test = model.predict_proba(x_test)[:, 1]
        ypred_bin_test = model.predict(x_test)

        # Create a copy of the test set and add columns for predicted probability and binary output
        scored_data = pd.DataFrame()
        scored_data[config["output"]["prob_output"]] = ypred_proba_test
        scored_data[config["output"]["bin_output"]] = ypred_bin_test
        scored_data[config["get_target"]["target"]] = y_test

        logger.info("Model scored on the test dataset with the following configuration: %s", config)
    except Exception as e:
        logger.error("Error occurred while scoring the model: %s", e)
        logger.debug("Exception details: %s", e)
        raise

    return scored_data


def save_scores(data: pd.DataFrame, path: Path) -> None:
    """
    Save the scored test dataset to disk.

    Args:
        data: DataFrame containing the scored test dataset.
        path: Path to the file where the dataset should be saved.
    """
    try:
        logger.debug("Preparing to save scored test dataset")
        data.to_csv(path, index=False)
        logger.info("Scored test dataset saved to %s", path)
    except FileNotFoundError:
        logger.error("Please provide a valid file location to save scored dataset to.")
        raise
    except Exception as e:
        logger.error("Error occurred while trying to write scored dataset to file: %s", e)
        raise