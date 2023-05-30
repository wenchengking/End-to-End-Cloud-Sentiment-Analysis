import logging
import pandas as pd
import boto3
import botocore
import sys
import io

logger = logging.getLogger(__name__)

def extract_clean_data(config: dict) -> pd.DataFrame:
    """read the cleaned .csv file from s3 bucket and return dataframe"""
    df = pd.DataFrame()
    # start a boto3 session
    try:
        session = boto3.Session()
        s3 = session.resource('s3')
        logger.info('s3 client created')
    except (botocore.exceptions.ProfileNotFound, botocore.exceptions.PartialCredentialsError) as e:
        logger.error('Profile/credentials error: %s', e)
        return df
    # Connect to S3
    s3 = boto3.resource('s3')
    # locate the bucket
    bucket = s3.Bucket(config['bucket_name'])
    clean_data = list(bucket.objects.filter(Prefix=config['input_prefix']))
    # if we have no files in the folder, exit
    if not clean_data:
        logger.error("The folder is empty.")
        sys.exit(1)
    else:
        # has file(s) in the folder
        if len(clean_data) >1: 
            # if we have more than one file in the folder
            logger.warning("The folder contains more than one one file.")
            # use the latest file 
            input_file = max(clean_data, key=lambda x: x.last_modified)
            input_file_key = input_file.key
            logging.info("Using latest file %s for training", input_file_key)
        else: 
            input_file_key = clean_data[0].key
            logging.info("Using file %s for training", input_file_key)
    # read dataframe from csv file
    obj = bucket.Object(input_file_key)
    data = obj.get()['Body'].read().decode(config['decode'])
    df = pd.read_csv(io.StringIO(data))
    logger.info("Cleaned Data Extracted Successfully")
    return df