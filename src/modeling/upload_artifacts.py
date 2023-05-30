import logging
from pathlib import Path
import os
import boto3
import botocore

logger = logging.getLogger(__name__)

def upload_artifacts(artifacts_path: Path, config: dict) -> list[str]:
    """Upload all the artifacts in the specified directory to S3
    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        config: Config required to upload artifacts to S3; see example config file for structure
    Returns:
        List of S3 uri's for each file that was uploaded
    """
    try:
        session = boto3.Session()
        s3_session = session.client('s3')
        logger.info('s3 client created')
    except (botocore.exceptions.ProfileNotFound, botocore.exceptions.PartialCredentialsError) as e:
        logger.error('Profile/credentials error: %s', e)
        return False
    bucket_name = os.getenv("BUCKET_NAME", config["bucket_name"])
    prefix = config["prefix"]

    # Upload each artifact file to S3
    for file_path in artifacts_path.iterdir():
        if file_path.is_file():
            key = f"{prefix}/{file_path.name}"
            try:
                s3_session.upload_file(str(file_path), bucket_name, key)
            except botocore.exceptions.ProfileNotFound as exc:
                logger.error("Profile/credentials error: %s", exc)
                return False
            s3_uri = f"s3://{bucket_name}/{key}"
            print(f"Successfully uploaded {file_path.name} to {bucket_name}/{key}")
            logger.info("Uploaded artifact %s to %s", file_path, s3_uri)
        else:
            logger.warning("Skipping non-file artifact %s", file_path)

    return True
