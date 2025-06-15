# inventory.py

import pandas as pd
import numpy as np
from scipy.stats import norm
import math

def calculate_inventory_metrics(forecast_df: pd.DataFrame, product_price: float, demand_std: float, config: dict) -> dict:
    """
    Calculates key inventory management metrics based on the forecast.

    Args:
        forecast_df (pd.DataFrame): The future forecast data.
        product_price (float): The unit price of the product.
        demand_std (float): The standard deviation of historical demand.
        config (dict): The configuration dictionary.

    Returns:
        dict: A dictionary of calculated inventory metrics.
    """
    inv_conf = config['inventory']
    avg_fcst = forecast_df['yhat'].mean()
    avg_lead_time, moq = inv_conf['avg_lead_time_days'], inv_conf['moq']

    lead_time_demand = avg_fcst * avg_lead_time
    z_score = norm.ppf(inv_conf['service_level'])
    safety_stock = z_score * demand_std * np.sqrt(avg_lead_time)
    reorder_point = lead_time_demand + safety_stock

    H = inv_conf['holding_cost_percentage'] * product_price if product_price and not pd.isna(product_price) else 0
    S = inv_conf['ordering_cost_percentage'] * product_price if product_price and not pd.isna(product_price) else 0

    D_annual = avg_fcst * 365
    eoq = math.sqrt((2 * D_annual * S) / H) if H > 0 else 0
    validated_eoq = max(eoq, moq)

    return {
        'avg_lead_time': avg_lead_time,
        'lead_time_demand': f"{lead_time_demand:,.0f}",
        'safety_stock': f"{safety_stock:,.0f}",
        'reorder_point': f"{reorder_point:,.0f}",
        'validated_eoq': f"{validated_eoq:,.0f}",
        'holding_cost': f"{H * D_annual:,.2f}",
        'ordering_cost': f"{S:,.2f}"
    }
