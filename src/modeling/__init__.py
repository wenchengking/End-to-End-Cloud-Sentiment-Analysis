import argparse
import datetime
import logging.config
from pathlib import Path

import evaluate_model as ep
import extract_clean_data as ecd
import score_model as sm
import train_model as trainm
import tune_model as tunem
import upload_artifacts as ua
import yaml

logging.config.fileConfig("../config/local.conf")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for classification model"
    )
    parser.add_argument(
        "--config", default="../config/model_config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error while loading configuration from %s: %s", args.config, str(e))
        else:
            logger.info("Configuration file loaded from %s", args.config)

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "artifacts")) / str(now)
    artifacts.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Acquire data from online repository and save to disk
    # Acquire data from online repository and save to disk
    clean_data = ecd.extract_clean_data(config["data_source"])

    # Split data into train/test set and train model based on config; save each to disk
    tmo, train, test = tunem.tune_model(clean_data, config["grid_search"])
    tunem.save_data(train, test, artifacts)
    tunem.save_model(tmo, artifacts / "tune_model_object.pkl")

    # Score model on test set; save scores to disk
    scores = sm.score_model(test, tmo, config["score_model"])
    sm.save_scores(scores, artifacts / "scores.csv")

    # Evaluate model performance metrics; save metrics to disk
    metrics = ep.evaluate_performance(scores, config["evaluate_performance"])
    ep.save_metrics(metrics, artifacts / "metrics.yaml")

    # Retrain model, and save to disk
    vectorizer, retrain_model = trainm.retrain_model(tmo,clean_data,config["train_model"])
    trainm.save_model(retrain_model, artifacts / "inference_model.pkl")
    trainm.save_model(vectorizer, artifacts / "vectorizer.pkl")

    ua.upload_artifacts(artifacts, config['aws_upload'])

