import pandas as pd
import numpy as np

# We will need the feature lists defined in your training script
LSTM_FEATURES = ['y', 'store_count', 'was_stocked_out', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'is_on_promotion', 'discount_percentage', 'is_month_start', 'is_month_end', 'sales_lag_1d', 'sales_lag_7d', 'sales_lag_14d', 'sales_lag_30d', 'rolling_avg_sales_7d', 'rolling_std_sales_7d']
XGB_FEATURES = [col for col in LSTM_FEATURES if col != 'y']

def create_future_features(history_df, days_to_forecast, store_growth_df):
    """
    Creates a future dataframe with all the necessary features for prediction.
    """
    last_date = history_df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast)
    future_df = pd.DataFrame({'ds': future_dates})

    # --- Create simple date-based and external features first ---
    future_df = future_df.merge(store_growth_df[['created_at', 'store_count']], left_on='ds', right_on='created_at', how='left').drop(columns='created_at')
    
    # FIX for Warning 1: Use .ffill() and .bfill() directly
    future_df['store_count'] = future_df['store_count'].ffill().bfill()
    
    future_df['was_stocked_out'] = 0
    future_df['is_on_promotion'] = 0
    future_df['discount_percentage'] = 0
    future_df['is_month_start'] = future_df['ds'].dt.is_month_start.astype(int)
    future_df['is_month_end'] = future_df['ds'].dt.is_month_end.astype(int)
    future_df['day_of_week_sin'] = np.sin(2 * np.pi * future_df['ds'].dt.dayofweek / 7)
    future_df['day_of_week_cos'] = np.cos(2 * np.pi * future_df['ds'].dt.dayofweek / 7)
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['ds'].dt.month / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['ds'].dt.month / 12)

    # We now combine the history with the future_df that *already has* our simple features.
    temp_history = pd.concat([history_df, future_df], ignore_index=True)
    temp_history.sort_values('ds', inplace=True)
    
    # Now create lag and rolling features on the combined dataframe
    for lag in [1, 7, 14, 30]:
        temp_history[f'sales_lag_{lag}d'] = temp_history['y'].shift(lag)
    temp_history['rolling_avg_sales_7d'] = temp_history['y'].shift(1).rolling(window=7, min_periods=1).mean()
    temp_history['rolling_std_sales_7d'] = temp_history['y'].shift(1).rolling(window=7, min_periods=1).std()
    
    # Get the future part of the dataframe which now contains ALL features
    future_df_with_features = temp_history[temp_history['ds'].isin(future_dates)].copy()
    
    # FIX for Warning 2: Explicitly infer dtypes after filling
    future_df_with_features.fillna(0, inplace=True)
    future_df_with_features = future_df_with_features.infer_objects(copy=False)

    return future_df_with_features

def predict_lstm(model_artifacts, history_df, days_to_forecast, store_growth_df):
    """Generates a forecast using a trained LSTM model."""
    final_lstm_model = model_artifacts['lstm']
    feature_scaler = model_artifacts['feature_scaler']
    target_scaler = model_artifacts['target_scaler']
    n_steps = final_lstm_model.input_shape[1]

    # Create a copy of history to append predictions to
    recursive_history_df = history_df.copy()
    predictions = []

    for _ in range(days_to_forecast):
        # Scale the last n_steps of data
        input_features_scaled = feature_scaler.transform(recursive_history_df[LSTM_FEATURES].tail(n_steps))
        input_seq = input_features_scaled.reshape(1, n_steps, len(LSTM_FEATURES))
        
        # Predict the next step
        pred_scaled = final_lstm_model.predict(input_seq, verbose=0)[0][0]
        
        # Inverse transform to get the actual value
        next_y_pred = target_scaler.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(next_y_pred)
        
        # Create the next row for the history dataframe to continue the recursion
        future_row_features = create_future_features(recursive_history_df, 1, store_growth_df)
        future_row_features['y'] = next_y_pred # Set the predicted 'y'
        
        recursive_history_df = pd.concat([recursive_history_df, future_row_features], ignore_index=True)

    return predictions

def predict_xgboost(model_artifacts, history_df, days_to_forecast, store_growth_df):
    """Generates a forecast using a trained XGBoost model."""
    model = model_artifacts['xgb']
    future_df = create_future_features(history_df, days_to_forecast, store_growth_df)
    predictions = model.predict(future_df[XGB_FEATURES])
    return predictions.tolist()

def predict_prophet(model_artifacts, history_df, days_to_forecast, store_growth_df):
    """Generates a forecast using a trained Prophet model."""
    model = model_artifacts['prophet']
    future_df = create_future_features(history_df, days_to_forecast, store_growth_df)
    # Ensure all required regressor columns are present for Prophet
    for regressor in model.extra_regressors:
        if regressor not in future_df.columns:
            raise ValueError(f"CRITICAL ERROR in future_df creation: Regressor '{regressor}' is missing.")
            
    forecast = model.predict(future_df)
    return forecast['yhat'].tolist()
