# import logging
import pandas as pd
import logging
import sys
import boto3
import botocore

logger = logging.getLogger(__name__)

# TODO: add docstring

def convert_ratings(data: pd.DataFrame, config: dict) -> pd.DataFrame:
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
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    logger.info("Downsampled data")
    return df_

def process_data(data:pd.DataFrame, config: dict) -> pd.DataFrame:
    conversion_df = convert_ratings(data, config)
    stratify_column = config['stratify_column']
    n_sample = int(float(config['downsample_ratio']) * conversion_df.shape[0])
    sampled_df = stratified_sample_df(conversion_df, stratify_column, n_sample)
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
    bucket = config['bucket_name']
    prefix = config['output_prefix']
    s3_key = f'{prefix}/{file_name}'
    
    try:
        response = s3_client.upload_file(file_name, bucket, s3_key)
    except boto3.exceptions.S3UploadFailedError as e:
        logger.error('Upload failed: %s', e)
        sys.exit(1)
    
    s3_address = f's3://{bucket}/{s3_key}'
    logging.info('File uploaded to %s', s3_address)
    return