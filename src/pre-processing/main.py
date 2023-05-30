import logging
import logging.config
import pandas as pd
import boto3
import botocore
import sys
import io
import os
import datetime

# set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# configure envionrment variables
config = {}
config['s3'], config['data_processing'] = {}, {}

config['s3']['decode'] = os.getenv('DECODE', 'utf-8') # we have an emoji
config['s3']['input_bucket'] = os.getenv('INPUT_BUCKET', 'msia423-group8-raw')
config['s3']['output_bucket'] = os.getenv('OUTPUT_BUCKET', 'msia423-group8-processed')
config['s3']['output_file_name'] = os.getenv('OUTPUT_FILE_NAME', 'cleaned_data.csv')

config['data_processing']['satisfied_threshold'] = os.getenv('SATISFIED_THRESHOLD', 3)
config['data_processing']['downsample_ratio'] = os.getenv('DOWNSAMPLE_RATIO', 0.05)
config['data_processing']['random_state'] = os.getenv('RANDOM_STATE', 42)
config['data_processing']['stratify_column'] = os.getenv('STRATIFY_COLUMN', 'Rating')

# functions
def read_from_s3(config: dict) -> pd.DataFrame:
    """read the .csv file from s3 bucket and return datafram"""
    df = pd.DataFrame()
    # start a boto3 session
    try:
        session = boto3.Session()
        s3 = session.resource('s3')
        logger.info('s3 client created')
    except (botocore.exceptions.ProfileNotFound, botocore.exceptions.PartialCredentialsError) as e:
        logger.error('Profile/credentials error: %s', e)
        return df
    # locate the bucket
    bucket = s3.Bucket(config['input_bucket'])
    raw_data = list(bucket.objects.all())
    # if we have no files in the folder, exit
    if not raw_data: 
        logger.error("The folder is empty.")
        sys.exit(1)
    else: 
        # has file(s) in the folder
        if len(raw_data) >1:
            # if we have more than one file in the folder
            logger.warning("The folder contains more than one file.")
            # use the latest file
            input_file = max(raw_data, key=lambda x: x.last_modified)
            input_file_key = input_file.key
            logger.info("Using latest file %s for processing", input_file_key)
        else: 
            input_file_key = raw_data[0].key
            logger.info("Using file %s for processing", input_file_key)
    # read dataframe from csv file
    obj = bucket.Object(input_file_key)
    data = obj.get()['Body'].read().decode(config['decode']) # we have an emoji
    df = pd.read_csv(io.StringIO(data))
    logger.info("file processed successfully")
    logger.debug('file size %s', df.shape)
    return df

def convert_ratings(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """convert ratings from numbers to binary (satisfy/unsatisfy)"""
    satisfied_threshold = config['satisfied_threshold']
    rating_column_name = config['stratify_column']
    try: 
        data[rating_column_name] = data[rating_column_name].apply(lambda x: 0 if x < satisfied_threshold else 1)
    except KeyError as e:
        logger.error("No columns called \"Rating\" in the dataframe")
        sys.exit(1)
    logger.info("Converted Ratings to binary")
    return data

def stratified_sample_df(df, col, n_samples):
    """create stratified sample from dataframe"""
    grouped = df.groupby(col)
    sampled_df = grouped.apply(lambda x: x.sample(n=n_samples, replace=True) if len(x) >= n_samples else x)
    clean_sampled_df = sampled_df.reset_index(drop=True)
    return clean_sampled_df

def process_data(data:pd.DataFrame, config: dict) -> pd.DataFrame:
    """process data: stratified downsample and convert ratings"""
    conversion_df = convert_ratings(data, config)
    stratify_column = config['stratify_column']
    n_sample = int(float(config['downsample_ratio']) * conversion_df.shape[0])
    logger.debug('downsample ratio %s', float(config['downsample_ratio']))
    logger.debug('downsample from %s to %s', conversion_df.shape[0], n_sample)
    sampled_df = stratified_sample_df(conversion_df, stratify_column, n_sample) # TODO: check the function for this
    logger.info("Finished processing data")
    return sampled_df

def write_to_s3(data:pd.DataFrame, config: dict) -> None:
    """upload single artifact to specified directory to S3"""
    # check for AWS credentials
    try:
        session = boto3.Session()
        s3_client = session.client('s3')
        logger.info('s3 client created')
    except (botocore.exceptions.ProfileNotFound, botocore.exceptions.PartialCredentialsError) as e:
        logger.error('Profile/credentials error: %s', e)
        sys.exit(1)
    # check if the bucket exists
    bucket = config['output_bucket']
    try:
        s3_client.head_bucket(Bucket=bucket)
        logger.info("Bucket '%s' exists", bucket)
    except Exception as e:
        logger.error(
            "The bucket '%s' does not exist or you do not have permission to access it", bucket
        )
        logger.error(e)
        sys.exit(1)
    # ouput to cvs
    folder = '/tmp/' # only /tmp is writable in AWS Lambda
    file_name_prefix, file_type = config['output_file_name'].split('.')
    now = int(datetime.datetime.now().timestamp())
    file_name = f"{file_name_prefix}_{now}.{file_type}"
    full_file_path = f"{folder}{file_name}"
    data.to_csv(full_file_path, index=False)
    logger.info('File saved to %s', full_file_path)
    # define s3 key
    s3_key = f'{file_name}'
    
    try:
        response = s3_client.upload_file(full_file_path, bucket, s3_key)
    except boto3.exceptions.S3UploadFailedError as e:
        logger.error('Upload failed: %s', e)
        sys.exit(1)
    
    s3_address = f's3://{bucket}/{file_name}'
    logging.info('File uploaded to %s', s3_address)
    return

# lambda handler
def lambda_handler(event, context):
    # Acquire data from s3 bucket
    logging.debug(config['s3'])
    ad_df = read_from_s3(config["s3"])

    # Process data and write to s3 bucket
    logging.debug(config['data_processing'])
    dp_df = process_data(ad_df, config["data_processing"])
    write_to_s3(dp_df, config["s3"])
    
if __name__ == "__main__":
    lambda_handler(None, None)