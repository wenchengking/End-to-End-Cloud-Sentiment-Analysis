import logging
import boto3
import botocore

logger = logging.getLogger(__name__)

def read_from_github(url: str, save_path: Path) -> None:
    raise NotImplementedError

def read_from_s3(artifacts: Path, config: dict) -> None:
    raise NotImplementedError

def move_to_archive(artifacts: Path, config: dict) -> None:
    # TODO: add folder name for raw and archive in .yaml
    raise NotImplementedError