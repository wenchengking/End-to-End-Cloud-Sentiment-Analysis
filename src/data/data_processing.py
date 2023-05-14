import logging
import pandas as pd

logger = logging.get_logger(__name__)

def convert_ratings(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    raise NotImplementedError

def downsample(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    raise NotImplementedError

def upload_to_s3(artifacts: Path, config: dict) -> None:
    raise NotImplementedError