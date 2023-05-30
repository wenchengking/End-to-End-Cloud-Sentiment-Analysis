# import logging
import pandas as pd
import logging
import sys
import boto3
import botocore

logger = logging.getLogger(__name__)

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
    # initialize s3 client
    try:
        session = boto3.Session(profile_name=config['user_profile'])
        s3_client = session.client('s3')
        logger.info('s3 client created')
    except (botocore.exceptions.ProfileNotFound, botocore.exceptions.PartialCredentialsError) as e:
        logger.error('Profile/credentials error: %s', e)
        sys.exit(1)
    # ouput to cvs
    file_name = config['output_file_name']
    data.to_csv(file_name, index=False)
    # define s3 key
    bucket = config['output_bucket']
    s3_key = f'{file_name}'
    
    try:
        response = s3_client.upload_file(file_name, bucket, s3_key)
    except boto3.exceptions.S3UploadFailedError as e:
        logger.error('Upload failed: %s', e)
        sys.exit(1)
    
    s3_address = f's3://{bucket}/{file_name}'
    logging.info('File uploaded to %s', s3_address)
    return