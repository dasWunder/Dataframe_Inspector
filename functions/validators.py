# functions/validators.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_dataframe_or_series(obj):
    if not isinstance(obj, (pd.DataFrame, pd.Series)):
        logger.error("Expected a pandas DataFrame or Series, got %s", type(obj))
        raise TypeError("Expected a pandas DataFrame or Series")

def validate_dataframe(obj):
    if not isinstance(obj, pd.DataFrame):
        logger.error("Expected a pandas DataFrame, got %s", type(obj))
        raise TypeError("Expected a pandas DataFrame")

def validate_positive_integer(n):
    if not isinstance(n, int) or n <= 0:
        logger.error("Invalid 'n' value: %s", n)
        raise ValueError("n must be an integer >= 1")
