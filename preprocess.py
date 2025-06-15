# preprocess.py

import pandas as pd
import numpy as np

def clean_numeric(value: any) -> float:
    """Cleans and converts a value to a float, handling various formats."""
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return float(value)
    value = str(value).strip()
    value = re.sub(r'[^\d\.-]', '', value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-series features for the model.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    df['discount_percentage'] = (df['disc'] / (df['revenue'] + df['disc'])).fillna(0) if (df['revenue'] + df['disc']).sum() > 0 else 0
    df['is_on_promotion'] = (df['disc'] > 0).astype(int)
    df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
    for lag in [1, 7, 14, 30]:
        df[f'sales_lag_{lag}d'] = df['y'].shift(lag).fillna(0)
    df['rolling_avg_sales_7d'] = df['y'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)
    df['rolling_std_sales_7d'] = df['y'].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['ds'].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['ds'].dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['ds'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['ds'].dt.month / 12)
    return df

def prepare_sku_data(sku_df: pd.DataFrame, store_growth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares and cleans the data for a single SKU.

    Args:
        sku_df (pd.DataFrame): The raw data for a single SKU.
        store_growth_df (pd.DataFrame): DataFrame with store growth information.

    Returns:
        pd.DataFrame: A cleaned and feature-engineered DataFrame ready for modeling.
    """
    # Merge store growth data
    sku_df = sku_df.merge(store_growth_df, on='created_at', how='left')
    sku_df['store_count'] = sku_df['store_count'].ffill().bfill()

    # Aggregate data by day
    agg_rules = {'qty': 'sum', 'store_count': 'first', 'was_stocked_out': 'max', 'disc': 'sum', 'revenue': 'sum'}
    model_df = sku_df.groupby('created_at').agg(agg_rules).reset_index()
    model_df.rename(columns={'created_at': 'ds', 'qty': 'y'}, inplace=True)

    # Outlier capping using IQR
    q1, q3 = model_df['y'].quantile([0.25, 0.75])
    iqr = q3 - q1
    model_df['y'] = model_df['y'].clip(lower=0, upper=q3 + 1.5 * iqr)

    # Fill missing dates and interpolate values
    if not model_df.empty:
        full_date_range = pd.date_range(start=model_df['ds'].min(), end=model_df['ds'].max(), freq='D')
        model_df = model_df.set_index('ds').reindex(full_date_range).reset_index().rename(columns={'index': 'ds'})
        model_df['y'] = model_df['y'].interpolate(method='linear').fillna(0)
        for col in ['store_count', 'was_stocked_out', 'disc', 'revenue']:
            model_df[col] = model_df[col].ffill().bfill().fillna(0)

    # Create time-series features
    model_df = create_features(model_df)
    model_df.sort_values('ds', inplace=True)
    
    return model_df
