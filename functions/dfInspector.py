# The module's goal is to study the data, preprocess the data and study (EDA). the module will contain various python functions to help achieving the goals

import pandas as pd
import logging
import matplotlib.pyplot as plt
from IPython.display import display

'''
Changelog
1. May 10, 2025. File creation, head_info development, some initial manual tests and test continuation will follow later
'''

# logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# the function which returns both head and info of a passed dataframe
# not only a dataframe, but a series object can be passed to the function as the argument
def head_info(df, n=5):
    """
    Displays the first n rows and metadata of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to inspect.
    n (int): Number of rows to display from the top (must be >= 0).
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Expected a pandas DataFrame, got %s", type(df))
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(n, int) or n < 0:
        logger.error("Invalid 'n' value: %s", n)
        raise ValueError("n must be a non-negative integer")

    logger.info("Displaying head with n=%d", n)
    display(df.head(n))
    df.info()