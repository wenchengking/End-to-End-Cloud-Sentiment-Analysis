import argparse
import yaml
import logging.config
import logging

import acquire_data as ad
import data_processing as dp

logging.config.fileConfig('logging_local.conf')
logger = logging.getLogger(__name__)
#TODO: logging file output?

if __name__ == "__main__":
    print(logger.name)
    # Parse command line arguments, default yaml file to data-config.yaml
    parser = argparse.ArgumentParser(description="Acquire data from s3 bucket, process and save to s3 bucket")
    parser.add_argument("--config", default="data-config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration file
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error(
                "Error while loading configuration from %s", args.config)
        else:
            logger.info("Configuration file loaded from %s", args.config)

    # Acquire data from s3 bucket
    ad_df = ad.read_from_s3(config["s3"])

    # Process data and write to s3 bucket
    dp_df = dp.process_data(ad_df, config["s3"])
    dp.write_to_s3(dp_df, config["s3"])