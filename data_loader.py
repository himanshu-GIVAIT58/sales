# data_loader.py

import pandas as pd
from typing import Optional

def load_csv_data(filepath: str, date_col: str, date_format: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    A generic function to load and perform initial date parsing on a CSV file.

    Args:
        filepath (str): The path to the CSV file.
        date_col (str): The name of the column containing dates.
        date_format (str, optional): The specific format of the date string. Defaults to None.

    Returns:
        Optional[pd.DataFrame]: A DataFrame if successful, otherwise None.
    """
    try:
        df = pd.read_csv(filepath)
        # Use a more robust date parsing that handles dayfirst automatically
        df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce', dayfirst=True).dt.normalize()
        df.dropna(subset=[date_col], inplace=True)
        return df
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {filepath}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load data from {filepath}. Reason: {e}")
        return None

def load_all_data(config: dict) -> tuple:
    """
    Loads all necessary data files for the forecasting pipeline.

    Args:
        config (dict): A dictionary containing file paths.

    Returns:
        tuple: A tuple containing sales_df, inventory_df, and holidays_df.
    """
    sales_df = load_csv_data(config['file_paths']['sales'], date_col='created_at')
    inventory_df = load_csv_data(config['file_paths']['inventory'], date_col='date')
    holidays_df = load_csv_data(config['file_paths']['holidays'], date_col='Date')
    
    if holidays_df is not None:
        holidays_df = holidays_df.rename(columns={'Date': 'ds', 'Name': 'holiday'})
        holidays_df = holidays_df[holidays_df['Type'].str.contains('|'.join(config['holiday_types']), na=False)]
        holidays_df = holidays_df[['ds', 'holiday']].drop_duplicates().dropna()

    return sales_df, inventory_df, holidays_df
