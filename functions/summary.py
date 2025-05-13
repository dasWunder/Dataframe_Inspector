# The module's goal is to study the data, preprocess the data and study (EDA). the module will contain various python functions to help achieving the goals

import pandas as pd
import logging
from IPython.display import display

# setting up printing options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

'''
Changelog
1. May 10, 2025. File creation, head_info development, some initial manual tests and test continuation will follow later
2. May 12, 2025. 
    - Spent a bit time to create a plan for future development, keeping in mind architecture and code quality
    - Added a test for Series object for head_info function
3. May 13, 2025.
'''


#####################################################################################################################################################
#########################################################DATAFRAME OVERVIEW##########################################################################
#####################################################################################################################################################

# logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# the function which returns both head and info of a passed dataframe
# not only a dataframe, but a series object can be passed to the function as the argument
def head_info(df, n=5):
    """
    Displays the first n rows and metadata of a DataFrame or Series.
    
    Args:
    df (pd.DataFrame or pd.Series): The data to inspect.
    n (int): Number of rows to display from the top (must be >= 0).
    """
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        logger.error("Expected a pandas DataFrame or Series, got %s", type(df))
        raise TypeError("df must be a pandas DataFrame or Series")
    
    if not isinstance(n, int) or n < 0:
        logger.error("Invalid 'n' value: %s", n)
        raise ValueError("n must be a non-negative integer")

    logger.info("Displaying head with n=%d", n)

    # Series: manually print metadata
    if isinstance(df, pd.Series):
        display(df.head(n))  # OK here
        print(f"\nSeries name: {df.name}")
        print(f"Type: {df.dtype}")
        return
    
    # DataFrame
    display(df.head(n))  # use print instead of display to avoid weird return rendering
    df.info()

''' To-do list:
First of all:
1. rework structure of folders
2. readme.md
3. learn logging and introduce it
4. the more the module grows, the more folders should be added. like __init__.py. separate part like summary, preprocess, eda and others should be as separate .py files


Development:
1. Test error messages for head_info
2. Add full_describe
3. Add numeric_describe
4. Add query function
5. Add various logic indices ways. Single condition, multiple conditions, various cols to output
6. Agg by function

#### Data preprocessing
1. missing values
2. unique values
3. duplicate number
4. duplicated rows
5. bring more ideas


#### EDA
1. Distribution
2. Boxplots
3. Correlation + heatmap
4. Scatter plots
5. Line plots
6. Bar plots (with either horizontal and vertical options)
7. Pairplot / jointplot
'''