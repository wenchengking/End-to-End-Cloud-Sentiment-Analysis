import logging
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

def self_train_test_split(df: pd.DataFrame, config: dict)->tuple:
    """Train Test Split.

    Args:
        df (pd.DataFrame): any dataset
        config (dict): A dictionary containing the configuration parameters for splitting.
    """
    X = df[config['feature_col']]
    y = df[config['target_col']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = config['test_size'], random_state = config['random_state'])
    return X_train, X_test, y_train, y_test

def transform(X_train: pd.DataFrame, X_test: pd.DataFrame)->tuple:
    """TF-IDF transformation.

    Args:
        X_train (pd.DataFrame): training dataset 
        X_test (pd.DataFrame): testing dataset
    """
    vectorizer= TfidfVectorizer()
    tf_x_train = vectorizer.fit_transform(X_train)
    tf_x_test = vectorizer.transform(X_test)
    return tf_x_train, tf_x_test

def tune_model(clean_data: pd.DataFrame, config:dict)->tuple:
    """Tune Both Models.

    Args:
        clean_data (pd.DataFrame): cleaned dataset
        config (dict): A dictionary containing the configuration parameters for tuning the model.
    """
    X_train, X_test, y_train, y_test=self_train_test_split(clean_data, config['train_test_split'])
    logger.info('Train Test Split Completed')
    tf_x_train, tf_x_test=transform(X_train, X_test)
    tf_x_train_df = pd.DataFrame.sparse.from_spmatrix(tf_x_train)
    tf_x_test_df = pd.DataFrame.sparse.from_spmatrix(tf_x_test)
    logger.info('Transformation Completed')

    # Instantiate models based on configuration
    logistic_model = LogisticRegression()
    naive_bayes_model = MultinomialNB()
    # Set up models for grid search
    #models = [logistic_model, naive_bayes_model]

    logistic_param_grid = config['logistic_param_grid']
    naive_bayes_param_grid = config['naive_bayes_param_grid']
    # Set up parameter grid for grid search
    #param_grid = [logistic_param_grid, naive_bayes_param_grid]

    logger.info('Logistic Grid Search Started')
    # Perform grid search for logistic regression
    logistic_CV = GridSearchCV(logistic_model, param_grid=logistic_param_grid, cv=config['cv'])
    logistic_CV.fit(tf_x_train, y_train)
    logger.info('Logistic Grid Search Ended')

    logger.info('Naive Bayes Grid Search Started')
    # Perform grid search for naive Bayes
    naive_bayes_CV = GridSearchCV(naive_bayes_model, param_grid=naive_bayes_param_grid, cv=config['cv'])
    naive_bayes_CV.fit(tf_x_train, y_train)
    logger.info('Naive Bayes Grid Search Ended')

    # Get the best parameters and scores for each model
    logistic_best_score = logistic_CV.best_score_
    naive_bayes_best_score = naive_bayes_CV.best_score_

    # Determine the best model
    best_model = None
    if logistic_best_score >= naive_bayes_best_score:
        best_model = logistic_model
    else:
        best_model = naive_bayes_model
    logger.info('Best Model Determined')

    best_model.fit(tf_x_train, y_train)
    
    y_train.index = tf_x_train_df.index
    y_test.index = tf_x_test_df.index

    return best_model, pd.concat([tf_x_train_df, y_train], axis=1), pd.concat([tf_x_test_df, y_test], axis=1)

def save_data(train: pd.DataFrame, test: pd.DataFrame, save_path: Path)->None:
    """Saves the train and test data as CSV files in the specified artifacts directory.

    Args:
        train (pd.DataFrame): A pandas DataFrame containing the train data to save.
        test (pd.DataFrame): A pandas DataFrame containing the test data to save.
        save_path (Path): A Path object representing the directory where the data should be saved.
    """
    train_path = save_path / "train.csv"
    test_path = save_path / "test.csv"

    train.to_csv(train_path, index=False)
    logger.info("Train data saved to %s.", train_path)

    test.to_csv(test_path, index=False)
    logger.info("Test data saved to %s.", test_path)

def save_model(tmo, save_path: Path)->None:
    """Saves the trained best model object to the specified file path.

    Args:
        model: The TMO to be saved.
        save_path (Path): A Path object representing the file path where the model should be saved.
    """
    try:
        with open(save_path, "wb") as file:
            pickle.dump(tmo, file)
        logger.info("TMO saved to %s", save_path)
    except FileNotFoundError:
        logger.error("Could not find file %s to save model.", save_path)
        sys.exit(1)
    except IsADirectoryError:
        logger.error("Cannot save model to a directory. Please provide a valid file path.")
        sys.exit(1)
