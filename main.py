import os
import sys
import logging
import pandas as pd
import numpy as np
import yaml
import concurrent.futures
from itertools import product
import warnings
import joblib

# Import from our custom modules
from data_loader import load_data
from preprocess import prepare_data, prepare_sku_specific_data
from forecasting_models import ProphetModel, LSTMModel, EnsembleModel, evaluate_model
from inventory import calculate_inventory_metrics

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def setup_logging(log_file: str):
    """Configures the logger to output to file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler
    try:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    except PermissionError:
        print(f"Error: No write permission for {log_file}. Logging to console only.")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Suppress verbose cmdstanpy messages
    cmdstanpy_logger = logging.getLogger('cmdstanpy')
    cmdstanpy_logger.setLevel(logging.WARNING)

def process_sku(args):
    """
    Processes a single SKU: trains models, evaluates, selects the best approach 
    (Ensemble or Prophet), forecasts, and calculates inventory metrics.
    """
    sku_id, sku_df, holidays_df, store_growth_df, config = args
    logging.info(f"--- Processing SKU: {sku_id} ---")
    
    sku_results = {'sku_id': sku_id}
    model_dir = config['paths']['model_directory']

    try:
        # 1. Prepare data specifically for this SKU
        model_df = prepare_sku_specific_data(sku_df)
        sku_results['last_processed_date'] = model_df['ds'].max().strftime('%Y-%m-%d')

        # 2. Data Validation and Splitting
        if len(model_df) < config['execution']['min_data_points_for_prophet']:
            raise ValueError(f"Insufficient data points: {len(model_df)}")

        train_df = model_df.iloc[:-config['execution']['validation_days']]
        validation_df = model_df.iloc[-config['execution']['validation_days']:]

        # 3. Model Training and Evaluation
        final_model_type = 'Prophet' # Default
        prophet_rmse, lstm_rmse = np.inf, np.inf

        # Train and evaluate Prophet
        prophet_model = ProphetModel(config, holidays_df).fit(train_df)
        prophet_rmse = evaluate_model(prophet_model, validation_df, 'prophet')
        
        # Conditionally train and evaluate LSTM
        use_lstm = len(model_df) >= config['execution']['min_data_points_for_lstm']
        if use_lstm:
            try:
                lstm_model = LSTMModel(config).fit(train_df)
                lstm_rmse = evaluate_model(lstm_model, validation_df, 'lstm', historical_df=train_df)
                
                # 4. Model Selection Logic
                if lstm_rmse < prophet_rmse * config['execution']['ensemble_fallback_threshold']:
                    final_model_type = 'Ensemble'
                else:
                    logging.info(f"LSTM performance ({lstm_rmse:.2f}) not superior to Prophet ({prophet_rmse:.2f}). Falling back to Prophet only.")
            except Exception as e:
                logging.error(f"LSTM model failed for SKU {sku_id}: {e}. Falling back to Prophet.")
        else:
            logging.info(f"Not enough data for LSTM ({len(model_df)} points). Using Prophet only.")

        sku_results['final_forecast_model_type'] = final_model_type
        
        # 5. Final Model Training and Forecasting
        logging.info(f"Retraining final model ('{final_model_type}') on all available data...")
        
        # Prepare future dataframe for forecasting
        future_dates = pd.date_range(start=model_df['ds'].max() + pd.Timedelta(days=1), periods=config['execution']['forecast_horizon_days'])
        future_df = pd.DataFrame({'ds': future_dates})
        future_df = pd.merge(future_df, store_growth_df, left_on='ds', right_on='created_at', how='left').drop(columns=['created_at'])
        future_df['store_count'] = future_df['store_count'].ffill().bfill()
        future_df['was_stocked_out'] = 0 # Assume no stockouts in the future
        future_df['weekday'] = future_df['ds'].dt.dayofweek

        # Retrain and predict with the chosen final model
        if final_model_type == 'Ensemble':
            prophet_final = ProphetModel(config, holidays_df).fit(model_df)
            lstm_final = LSTMModel(config).fit(model_df)
            final_model = EnsembleModel(config, prophet_final, lstm_final)
            final_forecast_df = final_model.predict(future_df, model_df)
        else: # Prophet Only
            final_model = ProphetModel(config, holidays_df).fit(model_df)
            final_forecast_df = final_model.predict(future_df)

        # 6. Post-processing and Inventory Calculation
        historical_max_sale = model_df['y'].max()
        upper_bound = max(historical_max_sale * 10, 50) # Cap forecast to a reasonable ceiling
        final_forecast_df['yhat'] = final_forecast_df['yhat'].clip(lower=0, upper=upper_bound)

        demand_std = model_df['y'].std() if model_df['y'].std() > 0 else 1.0
        product_price = sku_df['unit_price'].mean()
        inventory_metrics = calculate_inventory_metrics(final_forecast_df, product_price, demand_std, config)
        
        sku_results.update({
            'forecast_next_1_month': final_forecast_df.head(30)['yhat'].sum(),
            'forecast_next_3_months': final_forecast_df.head(90)['yhat'].sum(),
            'forecast_next_6_months': final_forecast_df['yhat'].sum(),
            **inventory_metrics
        })
        
        detailed_forecast = final_forecast_df[['ds', 'yhat']]
        detailed_forecast['sku_id'] = sku_id
        
        logging.info(f"Successfully processed SKU: {sku_id}")
        return sku_results, detailed_forecast

    except Exception as e:
        logging.error(f"SKU {sku_id} processing FAILED: {e}", exc_info=True)
        sku_results['error'] = str(e)
        return sku_results, pd.DataFrame()

def main():
    """Main execution script."""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config['paths']['log_file'])
    logging.info("--- Starting Forecasting Pipeline ---")

    # Create output directories
    os.makedirs(config['paths']['model_directory'], exist_ok=True)

    # Load and prepare data
    sales_df, inventory_df, holidays_df_raw = load_data(
        config['paths']['sales_data'],
        config['paths']['inventory_data'],
        config['paths']['holidays_data']
    )
    master_df, holidays_df, store_growth_df = prepare_data(sales_df, inventory_df, holidays_df_raw, config)

    # Identify SKUs to process
    all_skus = list(master_df['sku'].dropna().unique())
    if config['execution']['max_skus_to_process'] > 0:
        skus_to_process = all_skus[:config['execution']['max_skus_to_process']]
    else:
        skus_to_process = all_skus
        
    logging.info(f"Total SKUs found: {len(all_skus)}. Processing {len(skus_to_process)} SKUs.")

    tasks = [(sku, master_df[master_df['sku'] == sku].copy(), holidays_df, store_growth_df, config) for sku in skus_to_process]
    
    # Execute processing in parallel
    results_summary = []
    results_daily = []
    
    max_w = os.cpu_count() if config['execution']['max_workers'] == 0 else config['execution']['max_workers']

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as executor:
        for summary_result, daily_forecast in executor.map(process_sku, tasks):
            if summary_result:
                results_summary.append(summary_result)
            if not daily_forecast.empty:
                results_daily.append(daily_forecast)

    # Save results
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(config['paths']['output_summary'], index=False)
        logging.info(f"Saved summary results to {config['paths']['output_summary']}")

    if results_daily:
        daily_df = pd.concat(results_daily, ignore_index=True)
        daily_df.to_csv(config['paths']['output_daily'], index=False)
        logging.info(f"Saved daily forecast results to {config['paths']['output_daily']}")

    logging.info("--- Forecasting Pipeline Finished ---")

if __name__ == '__main__':
    main()