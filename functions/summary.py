''' Intro description
1. The module's goal is to study the data and get highlevel summary. 
2. The module will contain various python functions to help achieving the various initial data overview goals.
3. head(), info(), .dtypes, describe(), missing values, unique values, outliers (via quantiles)
'''
import pandas as pd
import logging
from functions.validators import validate_dataframe_or_series, validate_dataframe, validate_positive_integer
from IPython.display import display

# setting up printing options to read output conveniently
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def head_info(df: pd.DataFrame | pd.Series, n: int = 5) -> None:
    """
    Displays the first 5 rows (by default, else specified) and metadata of a DataFrame or Series.
    
    Args:
        df (pd.DataFrame or pd.Series): The object to inspect.
        n (int, optional): Number of rows to display. Must be non-negative and integer. Defaults to 5.

    Raises:
        TypeError: If `df` is not a pandas DataFrame or Series.
        ValueError: If `n` is negative and non integer.

    Returns:
        None
    """
    # validation for either a df or a series
    validate_dataframe_or_series(df)

    # validation for positive integer
    validate_positive_integer(n)
    
    # since series doesn't have method info(), this is a manual alternative to dataframe's info method
    if isinstance(df, pd.Series):
        display(df.head(n))
        print(f"Type: {df.dtype}")
        print(f"Missing values number: {df.isnull().sum()}")
        print(f'The shape is: {df.shape[0]}')
        return

    display(df.head(n))
    df.info()


def head(df: pd.DataFrame | pd.Series, n: int = 5) -> pd.DataFrame | pd.Series:
    """
    Returns the first `n` rows of a DataFrame or Series.
    
    Args:
        df (pd.DataFrame or pd.Series): The object to inspect.
        n (int, optional): Number of rows to return. Must be non-negative and integer. Defaults to 5.

    Raises:
        TypeError: If `df` is not a pandas DataFrame or Series.
        ValueError: If `n` is negative and non integer.

    Returns:
        pd.DataFrame or pd.Series: The top `n` rows.
    """
    
    # validation for either a df or a series
    validate_dataframe_or_series(df)

    # validation for positive integer
    validate_positive_integer(n)

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
    # validation for a df
    validate_dataframe(df)

    df.info()


def sample_rows(df: pd.DataFrame, n: int = 3, random_state: int | None = None) -> pd.DataFrame:
    """
    Returns a random sample of `n` rows from the DataFrame.

    Useful for quick manual inspection of the data quality and consistency.

    Args:
        df (pd.DataFrame): The DataFrame to sample from.
        n (int, optional): Number of rows to sample. Must be positive integer. Defaults to 3.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Raises:
        TypeError: If input is not a DataFrame.
        ValueError: If `n` is not a positive integer.

    Returns:
        pd.DataFrame: Sampled rows from the DataFrame.
    """
    # Validators
    validate_dataframe(df)
    validate_positive_integer(n)

    return df.sample(n=n, random_state=random_state)


def tail(df: pd.DataFrame | pd.Series, n: int = 5) -> pd.DataFrame | pd.Series:
    """
    Returns the last `n` rows of a DataFrame or Series.

    Args:
        df (pd.DataFrame or pd.Series): The object to inspect.
        n (int, optional): Number of rows to return from the bottom. Must be non-negative and integer. Defaults to 5.

    Raises:
        TypeError: If `df` is not a pandas DataFrame or Series.
        ValueError: If `n` is negative and non-integer.

    Returns:
        pd.DataFrame or pd.Series: The bottom `n` rows.
    """
    
    # validation for either a df or a series
    validate_dataframe_or_series(df)

    # validation for positive integer
    validate_positive_integer(n)

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
    # validation for a df
    validate_dataframe(df)

    if mode not in ('numerical', 'full'):
        logger.error("Invalid mode: %s", mode)
        raise ValueError("mode must be either 'numerical' or 'full'")

    return df.describe() if mode == 'numerical' else df.describe(include='all')


def columns_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing data types and number of unique values.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        pd.DataFrame: Summary of dtypes and unique values.
    """
    # validation for a df
    validate_dataframe(df)

    return pd.DataFrame({
        "dtype": df.dtypes,
        "unique_values_count": df.nunique()
    }).sort_values(by="unique_values_count", ascending=False)
    
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
        "total_size": df.size,
        "column_names": df.columns.tolist(),
        "missing_values_number": df.isnull().sum().sum(),
        "duplicated_rows": df.duplicated().sum()
    }

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns count and percentage of missing values per column.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        pd.DataFrame: Missing values summary.
    """
    missing_count = df.isnull().sum()
    missing_ratio_per_col = df.isnull().mean() * 100
    dtype = df.dtypes

    result = pd.DataFrame({
        "Column Name": df.columns,
        "Missing Count": missing_count,
        "Missing Ratio": missing_ratio_per_col,
        "Data Type": dtype
    }).reset_index(drop=True)

    return result

def top_values_summary(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Returns the top `n` most frequent values for each object or categorical column.

    The result includes value counts and percentages for quick overview of mode-level distribution.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        top_n (int, optional): Number of top values to return per column. Must be integer >= 1. Defaults to 3.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If top_n is not a positive integer.

    Returns:
        pd.DataFrame: DataFrame with columns: column, value, count, percentage.
    """
    # validators
    validate_dataframe(df)
    validate_positive_integer(top_n)

    summary = []

    for col in df.select_dtypes(include=["object", "category"]).columns:
        top_vals = df[col].value_counts().head(top_n)
        for val, count in top_vals.items():
            pct = count / df.shape[0] * 100
            summary.append({
                "column": col,
                "value": val,
                "count": count,
                "percentage": round(pct, 2)
            })

    return pd.DataFrame(summary).sort_values(by=["column", "count"], ascending=[True, False])



def duplicate_summary(df: pd.DataFrame) -> int:
    """
    Returns number of duplicated rows in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        int: Number of duplicated rows.
    """
    # validation for a df
    validate_dataframe(df)
    
    return df.duplicated().sum()

def outlier_summary(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Returns count of outliers per numeric column based on IQR.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        multiplier (float, optional): IQR multiplier. Defaults to 1.5.

    Returns:
        pd.DataFrame: Number of outliers per numeric column.
    """
    numeric_df = df.select_dtypes(include='number')
    outlier_counts = {}

    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outlier_counts[col] = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()

    return pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['outlier_count']).sort_values(by='outlier_count', ascending=False)


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
    print("\n ---Head---")
    display(df.head())
    
    print("\n ---Info---")
    print(df.info())
    
    print("\n ---Descriptive Stats---")
    display(describe(df, mode=describe_mode))
    
    print("\n ---Column Overview---")
    display(columns_overview(df))
    
    print("\n ---Shape Summary---")
    display(shape_summary(df))

    print("\n ---Missing Summary---")
    display(missing_summary(df))
    
    print("\n ---Top Values Summary---")
    display(top_values_summary(df))

    print("\n ---Duplicate Rows---")
    display(f"{duplicate_summary(df)} duplicated rows")
    
    print("\n ---Outliers Summary---")
    display(outlier_summary(df))
