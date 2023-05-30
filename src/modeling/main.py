import warnings
warnings.filterwarnings('ignore')

import datetime
import io
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import ast

import boto3
import botocore
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

import extract_clean_data as ecd
import score_model as sm
import evaluate_model as ep
import train_model as trainm
import tune_model as tunem
import upload_artifacts as ua

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# configure envrionment variables
config = {}

config['run_config'] = {}
config['run_config']['data_source'] = os.getenv('RC_DATA_SOURCE', 'S3')
# print('run_config:', config['run_config'])

config['data_source'] = {}
config['data_source']['bucket_name'] = os.getenv('DS_BUCKET_NAME', 'msia423-group8-processed')
config['data_source']['input_prefix'] = os.getenv('DS_INPUT_PREFIX', 'cleaned')
config['data_source']['decode'] = os.getenv('DS_DECODE', 'utf-8')
# print('data_source:', config['data_source'])

config['grid_search'] = {}

config['grid_search']['train_test_split'] = {}
config['grid_search']['train_test_split']['test_size'] = float(os.getenv('GS_TEST_SIZE', 0.2))
config['grid_search']['train_test_split']['random_state'] = int(os.getenv('GS_RANDOM_STATE', 42))
config['grid_search']['train_test_split']['target_col'] = os.getenv('GS_TARGET_COL', 'Rating')
config['grid_search']['train_test_split']['feature_col'] = os.getenv('GS_FEATURE_COL', 'Review')

config['grid_search']['cv'] = int(os.getenv('CV', 3))

config['grid_search']['logistic_model'] = os.getenv('GS_LOGISTIC_MODEL', 'LogisticRegression')
config['grid_search']['logistic_param_grid'] = {}
config['grid_search']['logistic_param_grid']['C'] = [float(elem) for elem in \
                                                            os.getenv('GS_PENALTY', "[0.2, 0.4, 0.8, 1.0, 1.2, 1.6]").strip('][').split(', ')]
config['grid_search']['logistic_param_grid']['penalty'] = os.getenv('GS_C', '[l1, l2]').strip('][').split(', ')

config['grid_search']['naive_bayes_model'] = os.getenv('GS_NAIVE_BAYES_MODEL', 'MultinomialNB')
config['grid_search']['naive_bayes_param_grid'] = {}
config['grid_search']['naive_bayes_param_grid']['alpha'] = [float(elem) for elem in \
                                                                os.getenv('GS_ALPHA', "[0.1, 0.5, 1.0]").strip('][').split(', ')]
config['grid_search']['naive_bayes_param_grid']['fit_prior'] = ast.literal_eval(os.getenv('GS_FIT_PRIOR', "[True, False]"))
# print('grid_search', config['grid_search'])

config['score_model'] = {}

config['score_model']['output'] = {}
config['score_model']['output']['prob_output'] = os.getenv('GS_PROB_OUTPUT', 'prob_predic_output')
config['score_model']['output']['bin_output'] = os.getenv('GS_BIN_OUTPUT', 'bin_predic_output')

config['score_model']['get_target'] = {}
config['score_model']['get_target']['target'] = os.getenv('GS_TARGET', 'Rating')
# print('score_model', config['score_model'])

config['evaluate_performance'] = {}
config['evaluate_performance']['features'] = {}
config['evaluate_performance']['features']['target'] = os.getenv('EP_TARGET', 'Rating')
config['evaluate_performance']['features']['predict_prob'] = os.getenv('EP_PREDICT_PROB', 'prob_predic_output')
config['evaluate_performance']['features']['predict_bin'] = os.getenv('EP_PREDICT_BIN', 'bin_predic_output')
config['evaluate_performance']['metrics'] = os.getenv('EP_METRICS', '[auc, confusion, accuracy, classification_report]').strip('][').split(', ')
# print('evaluate_performance', config['evaluate_performance'])

config['train_model'] = {}
config['train_model']['method'] = os.getenv('TM_METHOD', 'LogisticRegression')
config['train_model']['target_col'] = os.getenv('TM_TARGET_COL', 'Rating')
config['train_model']['feature_col'] = os.getenv('TM_FEATURE_COL', 'Review')
# print('train_model', config['train_model'])

config['aws_upload'] = {}
config['aws_upload']['upload'] = os.getenv('AW_UPLOAD', 'TRUE') == 'TRUE'
config['aws_upload']['bucket_name'] = os.getenv('AW_BUCKET_NAME', 'msia423-group8-artifact')
config['aws_upload']['prefix'] = os.getenv('AW_PREFIX', 'default')
config['aws_upload']['region'] = os.getenv('AW_REGION', 'us-east-2')
# print('aws_upload', config['aws_upload'])

def lambda_handler(event, context):
    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path("/tmp/model") / str(now) # on Lambda we don't need the timestamp, just for local comparison
    artifacts.mkdir(parents=True)

    # Acquire data from online repository and save to disk
    clean_data = ecd.extract_clean_data(config["data_source"])
    
    # Split data into train/test set and train model based on config; save each to disk
    tmo, train, test = tunem.tune_model(clean_data, config["grid_search"])
    tunem.save_data(train, test, artifacts)
    tunem.save_model(tmo, artifacts / "tune_model_object.pkl")

    # Score model on test set; save scores to disk
    scores = sm.score_model(test, tmo, config["score_model"])
    sm.save_scores(scores, artifacts / "scores.csv")
    print(scores.columns)

    # Evaluate model performance metrics; save metrics to disk
    metrics = ep.evaluate_performance(scores, config["evaluate_performance"])
    ep.save_metrics(metrics, artifacts / "metrics.yaml")

    # Retrain model, and save to disk
    vectorizer, retrain_model = trainm.retrain_model(tmo,clean_data,config["train_model"])
    trainm.save_model(retrain_model, artifacts / "inference_model.pkl")
    trainm.save_model(vectorizer, artifacts / "vectorizer.pkl")

    ua.upload_artifacts(artifacts, config['aws_upload'])