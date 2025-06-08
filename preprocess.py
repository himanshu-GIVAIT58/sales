import pandas as pd
import numpy as np
import re
import logging

def clean_numeric(value):
    """Clean string to extract numeric value."""
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return float(value)
    value = str(value).strip()
    value = re.sub(r'[^\d\.-]', '', value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def create_store_growth_features(df: pd.DataFrame, future_days: int) -> pd.DataFrame:
    """Creates a DataFrame with projected store counts for historical and future dates."""
    if 'created_at' not in df.columns or df['created_at'].isnull().all():
        logging.error("Cannot create store growth features: 'created_at' column is missing or empty.")
        # Return a simple dataframe to avoid crashing
        return pd.DataFrame({'created_at': pd.to_datetime([]), 'store_count': []})

    start_date = df['created_at'].min()
    end_date = df['created_at'].max()
    future_end_date = end_date + pd.DateOffset(days=future_days)
    full_date_range = pd.date_range(start_date, future_end_date, freq='D')
    
    store_growth_df = pd.DataFrame({'created_at': full_date_range})
    
    # Example growth logic (can be made more sophisticated)
    initial_stores, historical_growth, future_growth = 100, 5, 10
    
    store_growth_df['month_diff'] = ((store_growth_df['created_at'].dt.year - start_date.year) * 12 +
                                     (store_growth_df['created_at'].dt.month - start_date.month))
    store_growth_df['store_count'] = initial_stores + (store_growth_df['month_diff'] * historical_growth)
    
    future_mask = store_growth_df['created_at'] > end_date
    month_diff_future = ((store_growth_df.loc[future_mask, 'created_at'].dt.year - end_date.year) * 12 +
                         (store_growth_df.loc[future_mask, 'created_at'].dt.month - end_date.month))
    
    last_historical_count = store_growth_df.loc[~future_mask, 'store_count'].iloc[-1]
    store_growth_df.loc[future_mask, 'store_count'] = last_historical_count + (month_diff_future * future_growth)
    
    return store_growth_df[['created_at', 'store_count']]


def prepare_data(sales_df: pd.DataFrame, inventory_df: pd.DataFrame, holidays_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Orchestrates the full data preparation and feature engineering pipeline."""
    logging.info("Starting data preparation...")
    
    # Clean sales data
    for col in ['revenue', 'disc', 'qty']:
        if col in sales_df.columns:
            sales_df[col] = sales_df[col].apply(clean_numeric)
    sales_df.dropna(subset=['qty', 'revenue'], inplace=True)
    sales_df['created_at'] = pd.to_datetime(sales_df['created_at'], errors='coerce', dayfirst=True).dt.normalize()
    sales_df.dropna(subset=['created_at', 'sku'], inplace=True)

    # Prepare and merge inventory data
    inventory_df.rename(columns={'date': 'created_at'}, inplace=True)
    inventory_df['created_at'] = pd.to_datetime(inventory_df['created_at'], errors='coerce').dt.normalize()
    inventory_df.dropna(subset=['created_at', 'sku'], inplace=True)
    inventory_agg = inventory_df.groupby(['created_at', 'sku']).agg({'wh': 'sum'}).reset_index()
    sales_df = pd.merge(sales_df, inventory_agg, on=['created_at', 'sku'], how='left')
    sales_df['was_stocked_out'] = (sales_df['wh'] == 0).astype(int)
    sales_df['wh'] = sales_df['wh'].ffill().bfill().fillna(0)

    # Feature Engineering
    sales_df['unit_price'] = (sales_df['revenue'] / sales_df['qty']).replace([np.inf, -np.inf], np.nan)
    sales_df['unit_price'].fillna(sales_df.groupby('sku')['unit_price'].transform('median'), inplace=True)
    sales_df['unit_price'].fillna(sales_df['unit_price'].median(), inplace=True)
    
    sales_df['weekday'] = sales_df['created_at'].dt.dayofweek
    sales_df['is_weekend'] = sales_df['weekday'].isin([5, 6]).astype(int)

    # Prepare store growth features
    store_growth_df = create_store_growth_features(sales_df, config['execution']['forecast_horizon_days'])
    sales_df = pd.merge(sales_df, store_growth_df, on='created_at', how='left')
    sales_df['store_count'] = sales_df['store_count'].ffill().bfill()
    
    # Prepare holidays data
    holidays_df = holidays_df.rename(columns={'Date': 'ds', 'Name': 'holiday'})
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'], errors='coerce', dayfirst=True).dt.normalize()
    holidays_df = holidays_df[holidays_df['Type'].str.contains('|'.join(config['prophet']['holiday_types']), na=False)]
    holidays_df = holidays_df[['ds', 'holiday']].drop_duplicates().dropna()

    sales_df.sort_values('created_at', inplace=True)
    logging.info("Data preparation complete.")
    return sales_df, holidays_df, store_growth_df

def prepare_sku_specific_data(sku_df: pd.DataFrame) -> pd.DataFrame:
    """Prepares the final modeling-ready dataframe for a single SKU."""
    agg_rules = {
        'qty': 'sum', 
        'store_count': 'first', 
        'was_stocked_out': 'max', 
        'weekday': 'first'
    }
    
    # Aggregate data by day
    model_df = sku_df.groupby(sku_df['created_at']).agg(agg_rules).reset_index()
    model_df.rename(columns={'created_at': 'ds', 'qty': 'y'}, inplace=True)
    
    # Ensure all dates are present
    full_date_range = pd.date_range(start=model_df['ds'].min(), end=model_df['ds'].max(), freq='D')
    model_df = model_df.set_index('ds').reindex(full_date_range).reset_index().rename(columns={'index': 'ds'})
    
    # Forward-fill regressors and fill target with 0 for missing days
    model_df['store_count'] = model_df['store_count'].ffill()
    model_df['was_stocked_out'] = model_df['was_stocked_out'].fillna(0)
    model_df['weekday'] = model_df['ds'].dt.dayofweek
    model_df['y'] = model_df['y'].fillna(0)

    return model_df