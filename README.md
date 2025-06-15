# Multi-Model Sales Forecasting And Inventory Management Engine for SKU-Level Demand

## 1. Project Overview

This project implements a robust and automated forecasting engine designed to predict future sales demand at the individual Stock Keeping Unit (SKU) level. It employs a multi-model "champion-challenger" framework, where several advanced forecasting algorithms compete to produce the most accurate prediction for each SKU. The engine automates data preparation, feature engineering, model tuning, selection, and forecasting, providing not only sales predictions but also key inventory management metrics.

The core of the project is a Python script that systematically processes each SKU by:

1.  **Preparing and enriching** historical sales data.
2.  **Evaluating** a suite of models: Prophet, LSTM (with Monte Carlo Dropout), and XGBoost (with Quantile Regression).
3.  **Selecting the "champion" model** for each SKU based on backtesting performance (RMSE).
4.  **Handling edge cases** by using a Simple Moving Average (SMA) for SKUs with sporadic or insufficient data.
5.  **Training the final model** on all available data and generating a 183-day forecast with uncertainty intervals.
6.  **Calculating inventory metrics** like Safety Stock and Reorder Point.
7.  **Persisting** the forecasts, summary metrics, and trained models for future use.

The system is built for scalability and efficiency, utilizing parallel processing to handle a large number of SKUs simultaneously.

---

## 2. Project Architecture & Workflow

The forecasting process for each SKU follows a structured pipeline:

```
[Input Data] -> [Data Prep & Feature Engineering] -> [Model Evaluation & Selection] -> [Final Training & Forecasting] -> [Output & Persistence]
```

### Step 1: Environment and Configuration

* **Logging:** A comprehensive logging system is set up to record the process, warnings, and errors to both a file (`forecasting_log.txt`) and the console.
* **GPU Verification:** The script checks for and utilizes available GPUs to accelerate TensorFlow/Keras model training.
* **Configuration (`CONFIG`):** A central dictionary in the `if __name__ == '__main__':` block controls all major parameters, including file paths, model settings, and business logic thresholds.

### Step 2: Data Loading and Preparation

* **Input Files:** The engine requires three main CSV files:
    * `sales_data_complete__...csv`: Transactional sales data.
    * `query_result_...csv`: Inventory data to determine stockouts.
    * `indian_holidays.csv`: Holiday information for Prophet.
* **Data Cleaning:** Numeric columns are sanitized, and date columns are normalized.
* **Outlier Capping:** An Interquartile Range (IQR) method is used to cap extreme sales values, making the models more robust.
* **Feature Engineering:** A rich set of features is created to capture trends, seasonality, and other patterns:
    * **Time-Based Features:** `day_of_week`, `month`, `is_month_start/end` (encoded cyclically with sin/cos).
    * **Lag Features:** Sales from previous periods (1, 7, 14, 30 days ago).
    * **Rolling Window Features:** 7-day rolling average and standard deviation of sales.
    * **Exogenous Variables:** `store_count`, stockout flags, and promotion/discount information.

### Step 3: Model Evaluation and Champion Selection

For each SKU, the historical data is split into training and testing sets. Three sophisticated models are then tuned and evaluated.

1.  **Facebook Prophet:**
    * **Tuning:** A grid search over `changepoint_prior_scale` and `seasonality_prior_scale` is performed using cross-validation.
    * **Features:** Utilizes holidays and external regressors like `store_count` and promotion flags.
2.  **LSTM (Long Short-Term Memory Network):**
    * **Tuning:** Keras Tuner (`RandomSearch`) is used to find the optimal number of units, dropout rate, and learning rate.
    * **Uncertainty:** Monte Carlo Dropout is implemented during prediction to generate a distribution of possible outcomes, providing robust uncertainty intervals (`yhat_lower`, `yhat_upper`).
    * **Data Scaling:** `RobustScaler` is used to handle outliers in the feature set before feeding it to the network.
3.  **XGBoost (Extreme Gradient Boosting):**
    * **Tuning:** Optuna is used for efficient hyperparameter optimization to find the best `n_estimators`, `learning_rate`, `max_depth`, etc.
    * **Uncertainty:** The model is trained to predict specific quantiles (e.g., 10th, 50th, and 90th percentiles), providing a non-parametric estimate of uncertainty.

**Champion Selection:** The model with the lowest Root Mean Squared Error (RMSE) on the test set is declared the "champion" for that specific SKU.

### Step 4: Final Forecasting and Output

* **Final Training:** The champion model is re-trained on the *entire* historical dataset for the SKU.
* **Forecasting:** A 183-day (6-month) forecast is generated.
* **Inventory Metrics:** Key inventory parameters are calculated based on the forecast:
    * `Safety Stock`
    * `Reorder Point`
    * `Validated EOQ (Economic Order Quantity)`
* **Persistence:**
    * **Summary Results:** All metrics, hyperparameters, and forecast summaries are saved to `multi_sku_forecasting_final.csv`.
    * **Daily Forecasts:** The detailed day-by-day forecast is saved to `daily_forecasts_final.csv`.
    * **Trained Models:** The final trained model object (including scalers for LSTM) is saved as a `.pkl` file in the `saved_models/` directory for inference or analysis.

### Step 5: Handling Special Cases

* **Sporadic Data:** If a SKU's sales data consists of more than 60% zero-sales days (`sporadic_threshold`), the engine defaults to a 28-day Simple Moving Average (SMA) to provide a stable, baseline forecast.
* **Insufficient Data:** If a SKU has fewer than the minimum required data points, it is also handled by the SMA model.

## 3. How to Use

### Prerequisites

Ensure you have the required Python libraries installed:

```bash
pip install pandas numpy prophet xgboost tensorflow keras-tuner scikit-learn scipy optuna joblib
```

### Configuration

Before running the script, update the `CONFIG` dictionary and file paths in the main execution block (`if __name__ == '__main__':`).

```python
# --- Main Execution Block ---
if __name__ == '__main__':
    verify_gpu()
    CONFIG = {
        'max_skus_to_process': 30,  # Set to None to process all
        'max_workers': os.cpu_count(),
        'min_data_points_prophet': 30,
        'min_data_points_lstm': 60,
        'test_days': 30,
        'sporadic_threshold': 0.60,
        'sma_window': 28,
        # ... other parameters ...
    }

    # --- File Paths ---
    sales_file_path = "/path/to/your/sales_data.csv"
    inventory_file_path = "/path/to/your/inventory_data.csv"
    holidays_file_path = "/path/to/your/holidays.csv"
    output_csv_path = "/path/to/your/output/summary_forecast.csv"
    daily_forecast_output_path = "/path/to/your/output/daily_forecasts.csv"
    model_dir = '/path/to/your/saved_models'
```

### Execution

Run the script from your terminal:

```bash
python your_script_name.py
```

The script will log its progress to the console. Upon completion, the output files and saved models will be available in the specified directories. The script is designed to be incremental; it will check the `output_csv_path` and only retrain SKUs with new data since the last run.

## 4. Key Libraries and Dependencies

* **`pandas` & `numpy`**: Core data manipulation and numerical operations.
* **`prophet`**: For the Prophet forecasting model.
* **`xgboost`**: For the XGBoost regression model.
* **`tensorflow` & `keras`**: For building and training the LSTM model.
* **`keras-tuner`**: For hyperparameter tuning of the LSTM model.
* **`optuna`**: For hyperparameter tuning of the XGBoost model.
* **`scikit-learn`**: For preprocessing (`RobustScaler`) and metrics (`mean_squared_error`).
* **`joblib`**: For saving and loading the trained model objects.
* **`concurrent.futures`**: For parallel processing of SKUs.

![Screenshot 2025-06-08 074013](https://github.com/user-attachments/assets/abbb6e2f-0c85-4a5f-b595-1cf14fe4df16)

![Screenshot 2025-06-08 074026](https://github.com/user-attachments/assets/4f81a976-107b-4c46-92a1-2e4089379919)
  
