import numpy as np
import math
from scipy.stats import norm

def calculate_inventory_metrics(forecast_df: pd.DataFrame, product_price: float, demand_std: float, config: dict) -> dict:
    """Calculate standard inventory metrics based on forecast and config."""
    inv_conf = config['inventory']
    avg_fcst = forecast_df['yhat'].mean()
    
    # Use deterministic values from config
    avg_lead_time = inv_conf['avg_lead_time_days']
    moq = inv_conf['moq']
    
    lead_time_demand = avg_fcst * avg_lead_time
    
    # Calculate Safety Stock
    z_score = norm.ppf(inv_conf['service_level'])
    safety_stock = z_score * demand_std * np.sqrt(avg_lead_time)
    
    reorder_point = lead_time_demand + safety_stock
    
    # Calculate Economic Order Quantity (EOQ)
    H = inv_conf['holding_cost_percentage'] * product_price
    S = inv_conf['ordering_cost_percentage'] * product_price
    D_annual = avg_fcst * 365
    
    # Avoid division by zero
    eoq = math.sqrt((2 * D_annual * S) / H) if H > 0 and D_annual > 0 and S > 0 else 0
    
    # Ensure EOQ respects the Minimum Order Quantity
    validated_eoq = max(eoq, moq)
    
    return {
        'avg_lead_time': avg_lead_time,
        'lead_time_demand': lead_time_demand,
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'theoretical_eoq': eoq,
        'moq_used': moq,
        'validated_eoq': validated_eoq
    }