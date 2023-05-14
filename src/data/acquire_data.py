import logging
import pandas as pd
import boto3
import botocore
import sys
import io

logger = logging.getLogger(__name__)

#TODO: do we need to archive? 

def read_from_s3(config: dict) -> pd.DataFrame:
    """read the .csv file from s3 bucket and return datafram"""
    df = pd.DataFrame()
    # start a boto3 session
    try:
        session = boto3.Session(profile_name=config['user_profile'])
        s3 = session.resource('s3')
        logger.info('s3 client created')
    except (botocore.exceptions.ProfileNotFound, botocore.exceptions.PartialCredentialsError) as e:
        logger.error('Profile/credentials error: %s', e)
        return df
    # locate the bucket
    bucket = s3.Bucket(config['bucket_name'])
    raw_data = list(bucket.objects.filter(Prefix=config['input_prefix']))
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

if __name__ == "__main__":
    session = boto3.Session(profile_name='default')