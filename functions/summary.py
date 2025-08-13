# The module's goal is to study the data and get highlevel summary. 
# The module will contain various python functions to help achieving the various initial data overview goals.

import pandas as pd
import logging
from IPython.display import display

# setting up printing options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Logger setup
logger = logging.getLogger(__name__)
# changed logging from INFO to ERROR to minimize logging to necessary level
logger.setLevel(logging.ERROR)


def head_info(df: pd.DataFrame | pd.Series, n: int = 5) -> None:
    """
    Displays the first `n` rows and metadata of a DataFrame or Series.
    
    Args:
        df (pd.DataFrame or pd.Series): The object to inspect.
        n (int, optional): Number of rows to display. Must be non-negative. Defaults to 5.

    Raises:
        TypeError: If `df` is not a pandas DataFrame or Series.
        ValueError: If `n` is negative.

    Returns:
        None
    """
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        logger.error("Expected a pandas DataFrame or Series, got %s", type(df))
        raise TypeError("df must be a pandas DataFrame or Series")

    if not isinstance(n, int) or n <= 0:
        logger.error("Invalid 'n' value: %s", n)
        raise ValueError("n must be >= 1")

    if isinstance(df, pd.Series):
        display(df.head(n))
        print(f"\nSeries name: {df.name}")
        print(f"Type: {df.dtype}")
        return

    display(df.head(n))
    df.info()


def head(df: pd.DataFrame | pd.Series, n: int = 5) -> pd.DataFrame | pd.Series:
    """
    Returns the first `n` rows of a DataFrame or Series.
    
    Args:
        df (pd.DataFrame or pd.Series): The object to inspect.
        n (int, optional): Number of rows to return. Must be non-negative. Defaults to 5.

    Raises:
        TypeError: If `df` is not a pandas DataFrame or Series.
        ValueError: If `n` is negative.

    Returns:
        pd.DataFrame or pd.Series: The top `n` rows.
    """
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        logger.error("Expected a pandas DataFrame or Series, got %s", type(df))
        raise TypeError("df must be a pandas DataFrame or Series")

    if not isinstance(n, int) or n <= 0:
        logger.error("Invalid 'n' value: %s", n)
        raise ValueError("n must be >= 1")

    return df.head(n)


def info(df: pd.DataFrame) -> None:
    """
    Prints metadata of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.

    Returns:
        None
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Expected a pandas DataFrame, got %s", type(df))
        raise TypeError("df must be a pandas DataFrame")

    df.info()


def tail(df: pd.DataFrame | pd.Series, n: int = 5) -> pd.DataFrame | pd.Series:
    """
    Returns the last `n` rows of a DataFrame or Series.

    Args:
        df (pd.DataFrame or pd.Series): The object to inspect.
        n (int, optional): Number of rows to return from the bottom. Must be non-negative. Defaults to 5.

    Raises:
        TypeError: If `df` is not a pandas DataFrame or Series.
        ValueError: If `n` is negative.

    Returns:
        pd.DataFrame or pd.Series: The bottom `n` rows.
    """
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        logger.error("Expected a pandas DataFrame or Series, got %s", type(df))
        raise TypeError("df must be a pandas DataFrame or Series")

    if not isinstance(n, int) or n <= 0:
        logger.error("Invalid 'n' value: %s", n)
        raise ValueError("n must be >= 1")

    return df.tail(n)


def describe(df: pd.DataFrame, mode: str = 'numerical') -> pd.DataFrame:
    """
    Returns descriptive statistics for a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to describe.
        mode (str, optional): Either 'numerical' or 'full'. Defaults to 'numerical'.

    Raises:
        TypeError: If df is not a DataFrame.
        ValueError: If mode is not 'numerical' or 'full'.

    Returns:
        pd.DataFrame: Summary statistics.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Expected a pandas DataFrame, got %s", type(df))
        raise TypeError("df must be a pandas DataFrame")

    if mode not in ('numerical', 'full'):
        logger.error("Invalid mode: %s", mode)
        raise ValueError("mode must be either 'numerical' or 'full'")

    return df.describe() if mode == 'numerical' else df.describe(include='all')


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing data types and number of unique values.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        pd.DataFrame: Summary of dtypes and unique values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a DataFrame")

    return pd.DataFrame({
        "dtype": df.dtypes,
        "unique_values_count": df.nunique()
    }).sort_values(by="unique_values_count", ascending=False)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns count and percentage of missing values per column.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        pd.DataFrame: Missing values summary.
    """
    missing_count = df.isnull().sum()
    missing_pct = df.isnull().mean() * 100
    dtype = df.dtypes

    result = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "dtype": dtype
    })

    return result[result.missing_count > 0].sort_values(by="missing_pct", ascending=False)


def duplicate_summary(df: pd.DataFrame) -> int:
    """
    Returns number of duplicated rows in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        int: Number of duplicated rows.
    """
    return df.duplicated().sum()


def shape_summary(df: pd.DataFrame) -> dict:
    """
    Returns dictionary with number of rows, columns, and total values.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        dict: Basic shape info.
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total_values": df.size,
        "column_names": df.columns.tolist()
    }


def full_summary(df: pd.DataFrame, n: int = 5, describe_mode: str = 'numerical') -> None:
    """
    Prints a full summary overview of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        n (int, optional): Number of rows for head and tail. Defaults to 5.
        describe_mode (str, optional): 'numerical' or 'full'. Passed to get_description.

    Returns:
        None
    """
    print("\n🔹 Shape:")
    display(shape_summary(df))

    print("\n🔹 Column Overview:")
    display(column_overview(df))

    print("\n🔹 Missing Summary:")
    display(missing_summary(df))

    print("\n🔹 Duplicate Rows:")
    display(f"{duplicate_summary(df)} rows")

    print("\n🔹 Description:")
    display(describe(df, mode=describe_mode))

    print("\n🔹 Head:")
    display(df.head(n))

    print("\n🔹 Tail:")
    display(df.tail(n))
