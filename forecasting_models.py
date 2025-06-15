# forecasting_models.py

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping as XGBoostEarlyStopping
from keras_tuner import RandomSearch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import logging

def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculates common forecasting performance metrics."""
    # This function is duplicated here for model evaluation but could be moved to a shared utils file.
    from sklearn.metrics import mean_squared_error
    actual, predicted = np.array(actual), np.array(predicted)
    metrics = {'rmse': np.sqrt(mean_squared_error(actual, predicted))}
    non_zero_mask = actual != 0
    metrics['mape'] = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100 if np.any(non_zero_mask) else 0.0
    metrics['wape'] = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100 if np.sum(np.abs(actual)) > 0 else 0.0
    metrics['accuracy'] = 100 - metrics['wape']
    return metrics

def run_sma_forecast_with_ci(model_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Generates a forecast using SMA with a simple confidence interval."""
    sma_window = config['sma_window']
    test_pred = model_df['y'].shift(1).rolling(window=sma_window).mean().fillna(0)
    performance_metrics_val = evaluate_forecast(model_df['y'], test_pred)
    
    last_known_avg = model_df['y'].tail(sma_window).mean()
    std_dev = model_df['y'].tail(sma_window).std()
    
    z_score = 1.96 # for 95% confidence
    ci_margin = z_score * std_dev if pd.notna(std_dev) else last_known_avg * 0.2
    ci_lower = last_known_avg - ci_margin
    ci_upper = last_known_avg + ci_margin

    future_dates = pd.date_range(start=model_df['ds'].max() + pd.Timedelta(days=1), periods=183)
    final_forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': last_known_avg,
        'yhat_lower': max(0, ci_lower),
        'yhat_upper': ci_upper
    })
    return final_forecast_df, performance_metrics_val

def tune_prophet_params_optuna(train_df: pd.DataFrame, timeout: int) -> dict:
    """Tunes Prophet hyperparameters using Optuna."""
    # (Function content is the same as in your improved script)
    def objective(trial):
        params = {
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True),
            'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0, log=True),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
        }
        m = Prophet(**params).fit(train_df)
        df_cv = cross_validation(m, initial='180 days', period='30 days', horizon='30 days', parallel="threads", disable_diagnostics=True)
        df_p = performance_metrics(df_cv, rolling_window=1)
        return df_p['rmse'].values[0]
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, timeout=timeout, n_trials=15)
    return study.best_params

def tune_xgboost_params(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, sku_id: str) -> dict:
    """Tunes XGBoost hyperparameters using Optuna."""
    # (Function content is the same as in your improved script)
    def objective(trial):
        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42
        }
        model = XGBRegressor(**params)
        early_stopping_callback = XGBoostEarlyStopping(rounds=30, save_best=True)
        model.fit(train_df[features], train_df['y'], eval_set=[(test_df[features], test_df['y'])], callbacks=[early_stopping_callback], verbose=False)
        preds = model.predict(test_df[features])
        return np.sqrt(mean_squared_error(test_df['y'], preds))
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25, timeout=120)
    return study.best_params

def tune_lstm_model(X_train: np.ndarray, y_train: np.ndarray, n_features: int, n_steps: int, model_dir: str, sku_id: str, config: dict) -> Sequential:
    """Tunes LSTM hyperparameters using Keras Tuner."""
    # (Function content is the same as in your improved script)
    def build_model(hp):
        model = Sequential([
            Input(shape=(n_steps, n_features)),
            LSTM(units=hp.Int('units_1', 50, 200, step=50), activation='tanh', return_sequences=True),
            LSTM(units=hp.Int('units_2', 50, 200, step=50), activation='tanh'),
            Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')), loss='mse')
        return model
    try:
        tuner = RandomSearch(build_model, objective='val_loss', max_trials=5, executions_per_trial=1, directory=model_dir, project_name=f'lstm_tune_{sku_id}', overwrite=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        tuner.search(X_train, y_train, epochs=config['lstm']['epochs'], batch_size=config['lstm']['batch_size'], validation_split=0.2, callbacks=[early_stopping], verbose=0)
        best_model = tuner.get_best_models(num_models=1)[0]
        return best_model
    except Exception as e:
        logging.warning(f"SKU {sku_id}: LSTM tuning failed: {str(e)}. Falling back to default LSTM.")
        model = Sequential([Input(shape=(n_steps, n_features)), LSTM(100, activation='tanh', return_sequences=True), LSTM(100, activation='tanh'), Dropout(0.2), Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
        model.fit(X_train, y_train, epochs=config['lstm']['epochs'], batch_size=config['lstm']['batch_size'], validation_split=0.2, verbose=0)
        return model
