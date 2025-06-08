import pandas as pd
import sys
import logging

def load_data(sales_path: str, inventory_path: str, holidays_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads sales, inventory, and holidays data from CSV files."""
    try:
        sales_df = pd.read_csv(sales_path)
        inventory_df = pd.read_csv(inventory_path)
        holidays_df = pd.read_csv(holidays_path)
        logging.info("Successfully loaded all data files.")
        return sales_df, inventory_df, holidays_df
    except FileNotFoundError as e:
        logging.error(f"Error: File not found - {e.filename}")
        sys.exit(f"Critical Error: Could not find data file {e.filename}. Exiting.")