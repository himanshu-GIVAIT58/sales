import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Import our new prediction utility functions
from prediction_utils import predict_lstm, predict_xgboost, predict_prophet

# --- Basic Setup ---
# It's good practice for the API to have its own logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="SKU Demand Forecasting API",
    description="An API to serve forecasts for various SKUs using pre-trained models.",
    version="1.0.0"
)

# --- Model & Data Loading ---
# A dictionary to hold all our loaded models and their metadata
MODELS = {}
HISTORICAL_DATA = {}
STORE_GROWTH_DATA = pd.DataFrame()

@app.on_event("startup")
def load_models_and_data():
    """
    This function runs when the API server starts.
    It loads all trained model artifacts and necessary data into memory.
    """
    global STORE_GROWTH_DATA
    
    model_dir = './saved_models/'
    sales_file_path = "./home/jupyter/sales_data_complete___daily_drill_down_2025-05-29T12_37_43.113222731+05_30.csv"
    
    logger.info("--- API starting up: Loading all models and data ---")

    # 1. Load historical sales data (needed for feature creation)
    # In a production system, this would come from a database.
    try:
        sales_df = pd.read_csv(sales_file_path)
        sales_df['created_at'] = pd.to_datetime(sales_df['created_at'], errors='coerce', dayfirst=True).dt.normalize()
        sales_df.rename(columns={'created_at': 'ds', 'qty': 'y'}, inplace=True)
    except FileNotFoundError:
        logger.error(f"FATAL: Sales data file not found at {sales_file_path}")
        return

    # 2. Load and process Store Growth data
    # This logic is simplified from your training script
    start_date, end_date = sales_df['ds'].min(), sales_df['ds'].max()
    future_end_date = end_date + pd.DateOffset(days=365) # Load for at least a year
    full_date_range = pd.date_range(start_date, future_end_date, freq='D')
    STORE_GROWTH_DATA = pd.DataFrame({'created_at': full_date_range})
    STORE_GROWTH_DATA['month_diff'] = ((STORE_GROWTH_DATA['created_at'].dt.year - start_date.year) * 12 + (STORE_GROWTH_DATA['created_at'].dt.month - start_date.month))
    STORE_GROWTH_DATA['store_count'] = 100 + (STORE_GROWTH_DATA['month_diff'] * 5) # Simplified

    # 3. Scan model directory and load each model artifact
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory not found at {model_dir}. No models will be served.")
        return
        
    for filename in os.listdir(model_dir):
        if filename.startswith("model_") and filename.endswith(".pkl"):
            sku_id = filename.replace("model_", "").replace(".pkl", "")
            logger.info(f"Loading model for SKU: {sku_id}")
            try:
                MODELS[sku_id] = joblib.load(os.path.join(model_dir, filename))
                # Store historical data specific to this SKU
                HISTORICAL_DATA[sku_id] = sales_df[sales_df['sku'] == sku_id].copy()
            except Exception as e:
                logger.error(f"Failed to load model for SKU {sku_id}. Error: {e}")

    logger.info(f"--- Startup complete. {len(MODELS)} models loaded successfully. ---")


class ForecastRequest(BaseModel):
    sku_id: str
    days_to_forecast: int = 30 

class Forecast(BaseModel):
    date: str
    predicted_sales: float

class ForecastResponse(BaseModel):
    sku_id: str
    model_type: str
    forecast: list[Forecast]

@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"status": "Forecasting API is running", "models_loaded": list(MODELS.keys())}

@app.post("/predict/", response_model=ForecastResponse, tags=["Forecasting"])
def get_prediction(request: ForecastRequest):
    """
    Accepts a SKU ID and number of days, and returns the demand forecast.
    """
    sku_id = request.sku_id
    days = request.days_to_forecast

    logger.info(f"Received prediction request for SKU: {sku_id} for {days} days.")
  
    if sku_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found. No trained model available.")

    model_artifacts = MODELS[sku_id]
    history_df = HISTORICAL_DATA.get(sku_id)

    if history_df is None or history_df.empty:
        raise HTTPException(status_code=404, detail=f"Historical data for SKU '{sku_id}' not found.")

    # 2. Determine model type and call the appropriate prediction function
    model_type = "Unknown"
    predictions = []
    
    if 'lstm' in model_artifacts:
        model_type = "LSTM"
        predictions = predict_lstm(model_artifacts, history_df, days, STORE_GROWTH_DATA)
    elif 'xgb' in model_artifacts:
        model_type = "XGBoost"
        predictions = predict_xgboost(model_artifacts, history_df, days, STORE_GROWTH_DATA)
    elif 'prophet' in model_artifacts:
        model_type = "Prophet"
        predictions = predict_prophet(model_artifacts, history_df, days, STORE_GROWTH_DATA)
    else:
        # This can be expanded to include SMA or other models
        raise HTTPException(status_code=500, detail=f"No valid model found in artifact for SKU '{sku_id}'.")
        
    logger.info(f"Successfully generated forecast using {model_type} model.")

    # 3. Format the response
    last_date = history_df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    
    response_forecast = [
        {"date": date.strftime("%Y-%m-%d"), "predicted_sales": round(max(0, pred), 2)}
        for date, pred in zip(future_dates, predictions)
    ]
    
    return {
        "sku_id": sku_id,
        "model_type": model_type,
        "forecast": response_forecast
    }
