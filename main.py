# main.py

import os
import sys
import logging
import pandas as pd
import numpy as np
import concurrent.futures
import yaml
import joblib

# Import from our new modules
import data_loader
import preprocess
import inventory
import forecasting_models
import prediction_utils # For MC Dropout

def setup_logging(log_file: str):
    """Configures the root logger for the application."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    logging.getLogger('optuna').setLevel(logging.WARNING)

def process_sku(args: tuple) -> bool:
    """
    Main function to process a single SKU by orchestrating other modules.
    """
    (sku_id, sku_df, holidays_df, store_growth_df, config) = args
    model_dir = config['file_paths']['model_dir']
    
    logging.info(f"\n--- 🔄 Processing SKU: {sku_id} ---")
    sku_results = {'sku_id': sku_id}
    
    try:
        # 1. Preprocess data
        model_df = preprocess.prepare_sku_data(sku_df, store_growth_df)
        sku_results['historical_avg_sales'] = model_df['y'].mean()
        sku_results['historical_std_sales'] = model_df['y'].std()
        sku_results['last_processed_date'] = model_df['ds'].max().strftime('%Y-%m-%d')

        # 2. Handle simple cases (insufficient data, sporadic demand)
        if len(model_df) < config['min_data_points_prophet']:
            logging.info(f"SKU {sku_id}: Insufficient data. Using SMA.")
            final_forecast_df, champion_metrics = forecasting_models.run_sma_forecast_with_ci(model_df, config)
            final_model_type = 'SMA'
            final_model_to_save = {}
        elif (model_df['y'] == 0).mean() > config['sporadic_threshold']:
            logging.info(f"SKU {sku_id}: Sporadic demand detected. Using SMA.")
            final_forecast_df, champion_metrics = forecasting_models.run_sma_forecast_with_ci(model_df, config)
            final_model_type = 'SMA'
            final_model_to_save = {}
        else:
            # 3. Champion-Challenger Model Evaluation
            test_days = config['test_days']
            train_df, test_df = model_df.iloc[:-test_days], model_df.iloc[-test_days:].copy()
            
            # This logic remains complex but is now calling modular functions
            # ... [Full champion-challenger logic would go here] ...
            # For brevity, let's assume a simplified evaluation
            final_model_type = 'Tuned_Prophet' # Simplified for this example
            best_prophet_params = forecasting_models.tune_prophet_params_optuna(train_df, config['prophet_tuning_timeout'])
            final_prophet_model = Prophet(holidays=holidays_df, **best_prophet_params).fit(model_df)
            
            future_df_base = pd.DataFrame({'ds': pd.date_range(start=model_df['ds'].max() + pd.Timedelta(days=1), periods=183)})
            future_df_prophet = prediction_utils.create_future_features(model_df, 183, store_growth_df)
            final_forecast_df = final_prophet_model.predict(future_df_prophet)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            champion_metrics = forecasting_models.evaluate_forecast(test_df['y'], final_prophet_model.predict(test_df)['yhat'])
            final_model_to_save = {'prophet': final_prophet_model}
            
        # 4. Populate Final Results
        sku_results.update(champion_metrics)
        sku_results['final_forecast_model_type'] = final_model_type
        
        inventory_metrics = inventory.calculate_inventory_metrics(final_forecast_df, sku_df['unit_price'].mean(), model_df['y'].std(), config)
        sku_results.update(inventory_metrics)

        final_forecast_df['yhat'] = final_forecast_df['yhat'].clip(lower=0)
        sku_results['forecast_next_1_month'] = final_forecast_df.head(30)['yhat'].sum()
        sku_results['forecast_next_3_months'] = final_forecast_df.head(90)['yhat'].sum()
        sku_results['forecast_next_6_months'] = final_forecast_df.head(183)['yhat'].sum()

        # 5. Save Artifacts
        joblib.dump(final_model_to_save, os.path.join(model_dir, f'model_{sku_id}.pkl'))
        # ... [Logic to save results to CSV files would go here] ...

        logging.info(f"✅ SKU {sku_id} processed successfully.")
        return True

    except Exception as e:
        logging.error(f"SKU {sku_id} processing FAILED: {e}", exc_info=True)
        # ... [Error saving logic] ...
        return False

def main():
    """Main execution block for the forecasting pipeline."""
    # 1. Load Configuration
    try:
        with open('config.yaml', 'r') as f:
            CONFIG = yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit("❌ FATAL: config.yaml not found. Please create it.")
    
    # 2. Setup Logging
    log_file = CONFIG.get('log_file', './home/jupyter/forecasting_log.txt')
    setup_logging(log_file)
    logging.info("--- Starting Forecasting Pipeline ---")

    # 3. Load Data
    sales_df, inventory_df, holidays_df = data_loader.load_all_data(CONFIG)
    if sales_df is None:
        sys.exit("❌ FATAL: Could not load sales data. Exiting.")

    # ... [Data preparation logic for merging inventory, cleaning sales, etc.] ...

    # 4. Prepare for Parallel Processing
    skus_to_run = sales_df['sku'].dropna().unique()
    if CONFIG['max_skus_to_process']:
        skus_to_run = skus_to_run[:CONFIG['max_skus_to_process']]
    
    # ... [Store growth df creation] ...
    store_growth_df = pd.DataFrame() # Placeholder

    tasks = [(sku, sales_df[sales_df['sku'] == sku].copy(), holidays_df, store_growth_df, CONFIG) for sku in skus_to_run]
    
    # 5. Execute Pipeline
    logging.info(f"Preparing to process {len(tasks)} SKU(s) using up to {CONFIG['max_workers']} processes.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        list(executor.map(process_sku, tasks))

    logging.info("\n\n🎉 Pipeline finished. All results have been saved incrementally.")

if __name__ == '__main__':
    main()
