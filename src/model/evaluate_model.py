import logging

logger = logging.getLogger(__name__)

def evaluate_performance(scores: pd.DataFrame, config: dict) -> dict:
    raise NotImplementedError

def save_metrics(metrics: dict, save_path: Path) -> None:
    raise NotImplementedError