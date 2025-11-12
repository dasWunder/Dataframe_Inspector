# module with custom validators
# functions/validators.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# validators
def validate_dataframe_or_series(obj):
    """
    Validates that the input is a pandas DataFrame or Series.

    Args:
        obj (object): The object to validate.

    Raises:
        TypeError: If `obj` is not a DataFrame or Series.
    """
    if not isinstance(obj, (pd.DataFrame, pd.Series)):
        logger.error("Expected a pandas DataFrame or Series, got %s", type(obj))
        raise TypeError("Expected a pandas DataFrame or Series")

def validate_dataframe(obj):
    """
    Validates that the input is a pandas DataFrame.

    Args:
        obj (object): The object to validate.

    Raises:
        TypeError: If `obj` is not a DataFrame.
    """
    if not isinstance(obj, pd.DataFrame):
        logger.error("Expected a pandas DataFrame, got %s", type(obj))
        raise TypeError("Expected a pandas DataFrame")

def validate_positive_integer(n):
    """
    Validates that the input is a positive integer (>=1).

    Args:
        n (object): The value to validate.

    Raises:
        ValueError: If `n` is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        logger.error("Invalid 'n' value: %s", n)
        raise ValueError("n must be an integer >= 1")
