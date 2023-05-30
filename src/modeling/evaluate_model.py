import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yaml
import sklearn.metrics
import numpy as np

logger = logging.getLogger(__name__)

def evaluate_performance(scores: pd.DataFrame, config: dict) -> dict:
    """Evaluate the performance of the model using specified metrics

    Args:
        scores: DataFrame containing the true labels and predicted values
        config: Configuration dictionary containing evaluation metrics and column names

    Returns:
        A dictionary containing the calculated metrics
    """
    try:
        # Extract the necessary columns from the scores DataFrame
        true_label, pred_prob, pred_bin = extract_columns(scores, config)
    except KeyError as e:
        logger.exception("Column not found in scores DataFrame: %s", e)
        raise

    metrics = {}

    try:
        # Calculate the requested evaluation metrics
        if "auc" in config["metrics"]:
            metrics["auc"] = calculate_auc(true_label, pred_prob)
        if "confusion" in config["metrics"]:
            metrics["confusion"] = calculate_confusion_matrix(true_label, pred_bin)
        if "accuracy" in config["metrics"]:
            metrics["accuracy"] = calculate_accuracy(true_label, pred_bin)
        if "classification_report" in config["metrics"]:
            metrics["classification_report"] = calculate_classification_report(true_label, pred_bin)
    except Exception as e:
        logger.exception("Error occurred while calculating performance metrics: %s", e)
        raise

    logger.info("Performance evaluation completed successfully.")
    return metrics


def extract_columns(scores: pd.DataFrame, config: dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Extract the necessary columns from the scores DataFrame

    Args:
        scores: DataFrame containing the true labels and predicted values
        config: Configuration dictionary containing column names

    Returns:
        A tuple of three pandas Series containing the true labels, predicted probabilities, and predicted binary labels
    """
    try:
        true_label = scores[config["features"]["target"]]
        pred_prob = scores[config["features"]["predict_prob"]]
        pred_bin = scores[config["features"]["predict_bin"]]
    except KeyError as e:
        logger.error("Column not found in scores DataFrame: %s", e)
        logger.debug("Exception details: %s", e)
        raise

    logger.info("Relevant column extracted successfully.")
    return true_label, pred_prob, pred_bin


def calculate_auc(true_label: pd.Series, pred_prob: pd.Series) -> float:
    """Calculate the area under the receiver operating characteristic curve (ROC AUC)

    Args:
        true_label: pandas Series containing the true labels
        pred_prob: pandas Series containing the predicted probabilities

    Returns:
        The ROC AUC score as a float
    """
    return sklearn.metrics.roc_auc_score(true_label, pred_prob)


def calculate_confusion_matrix(true_label: pd.Series, pred_bin: pd.Series) -> List[List[int]]:
    """Calculate the confusion matrix

    Args:
        true_label: pandas Series containing the true labels
        pred_bin: pandas Series containing the predicted binary labels

    Returns:
        The confusion matrix as a list of lists
    """
    return sklearn.metrics.confusion_matrix(true_label, pred_bin).tolist()


def calculate_accuracy(true_label: pd.Series, pred_bin: pd.Series) -> float:
    """Calculate the accuracy

    Args:
        true_label: pandas Series containing the true labels
        pred_bin: pandas Series containing the predicted binary labels

    Returns:
        The accuracy score as a float
    """
    return sklearn.metrics.accuracy_score(true_label, pred_bin)


def calculate_classification_report(true_label: pd.Series, pred_bin: pd.Series) -> dict:
    """Calculate the classification report

    Args:
        true_label: pandas Series containing the true labels
        pred_bin: pandas Series containing the predicted binary labels

    Returns:
        The classification report as a dictionary
    """
    return sklearn.metrics.classification_report(true_label, pred_bin, output_dict=True)


def save_metrics(metrics: dict, file_path: Path) -> None:
    """Save the metrics to a file in YAML format

    Args:
        metrics: Dictionary containing the calculated performance metrics
        file_path: Path to the file where the metrics will be saved
    """
    converted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.int64, np.float64, np.bool_)):
            value = value.item()
        converted_metrics[key] = value

    try:
        logger.debug("Preparing to save metrics to file")
        # Write the metrics dictionary to a file in YAML format
        with file_path.open("w") as f:
            yaml.safe_dump(converted_metrics, f)
        logger.info("Metrics saved to %s", file_path)
    except yaml.error.YAMLError as e:
        logger.error("Error occurred while saving metrics to file: %s", e)
        logger.debug("Exception details: %s", e)
        raise