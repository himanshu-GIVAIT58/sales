import pandas as pd
import numpy as np
import logging
import os
import joblib
from prophet import Prophet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

# Suppress TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# --- Base Class (for structure) ---
class BaseModel:
    def __init__(self, config):
        self.model = None
        self.config = config

    def fit(self, df):
        raise NotImplementedError

    def predict(self, future_df):
        raise NotImplementedError

# --- Prophet Model ---
class ProphetModel(BaseModel):
    def __init__(self, config, holidays_df):
        super().__init__(config)
        self.holidays_df = holidays_df
        prophet_params = self.config['prophet'].copy()
        prophet_params.pop('holiday_types', None) # Remove non-Prophet arg
        self.model = Prophet(**prophet_params, holidays=self.holidays_df)
        self.model.add_regressor('store_count')
        self.model.add_regressor('was_stocked_out')
    
    def fit(self, df: pd.DataFrame):
        logging.info("Fitting Prophet model...")
        self.model.fit(df[['ds', 'y', 'store_count', 'was_stocked_out']])
        return self
        
    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Predicting with Prophet model...")
        return self.model.predict(future_df)

# --- LSTM Model ---
class LSTMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.scaler = RobustScaler()
        self.features = ['y', 'store_count', 'was_stocked_out', 'weekday']
        self.n_steps = self.config['lstm']['n_steps']
        self.n_features = len(self.features)
        self.model = self._build_model()

    def _build_model(self):
        """Builds the Keras LSTM model."""
        model = Sequential([
            Input(shape=(self.n_steps, self.n_features)),
            LSTM(100, activation='relu', return_sequences=True),
            Dropout(self.config['lstm']['dropout_rate']),
            LSTM(50, activation='relu'),
            Dropout(self.config['lstm']['dropout_rate']),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['lstm']['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def _create_sequences(self, data):
        """Creates input sequences (X) and target values (y) for the LSTM."""
        X, y = [], []
        for i in range(len(data) - self.n_steps):
            X.append(data[i:i + self.n_steps])
            y.append(data[i + self.n_steps, 0]) # Target is the 'y' column (first feature)
        return np.array(X), np.array(y)

    def fit(self, df: pd.DataFrame):
        """Fits the LSTM model."""
        logging.info("Fitting LSTM model...")
        df_scaled = self.scaler.fit_transform(df[self.features])
        X_train, y_train = self._create_sequences(df_scaled)
        
        if X_train.shape[0] == 0:
            logging.warning("Not enough data to create sequences for LSTM. Skipping fit.")
            return self

        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.model.fit(
            X_train, y_train,
            epochs=self.config['lstm']['epochs'],
            batch_size=self.config['lstm']['batch_size'],
            verbose=0,
            callbacks=[early_stopping]
        )
        return self

    def predict(self, future_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
        """Makes future predictions with the trained LSTM model."""
        logging.info("Predicting with LSTM model...")
        
        # Combine historical data for making predictions
        full_df = pd.concat([historical_df, future_df], ignore_index=True)
        
        # Scale all data
        scaled_data = self.scaler.transform(full_df[self.features])
        
        predictions = []
        # Use the last n_steps of historical data as the initial input
        current_batch = scaled_data[-self.config['execution']['forecast_horizon_days'] - self.n_steps:-self.config['execution']['forecast_horizon_days']].reshape(1, self.n_steps, self.n_features)

        for i in range(self.config['execution']['forecast_horizon_days']):
            # Predict the next step
            current_pred = self.model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred)
            
            # Get the features for the next time step from future_df
            next_features = scaled_data[-self.config['execution']['forecast_horizon_days'] + i, 1:]
            
            # Create the new input for the next prediction
            new_step = np.append(current_pred, next_features).reshape(1, 1, self.n_features)
            current_batch = np.append(current_batch[:, 1:, :], new_step, axis=1)

        # Inverse transform the predictions to the original scale
        # We need to create a dummy array with the same shape as the scaler expects
        dummy_array = np.zeros((len(predictions), self.n_features))
        dummy_array[:, 0] = np.array(predictions).flatten()
        yhat_lstm = self.scaler.inverse_transform(dummy_array)[:, 0]

        forecast_df = pd.DataFrame({'ds': future_df['ds'], 'yhat_lstm': yhat_lstm})
        return forecast_df
        
# --- Ensemble Model ---
class EnsembleModel(BaseModel):
    def __init__(self, config, prophet_model: ProphetModel, lstm_model: LSTMModel):
        super().__init__(config)
        self.prophet_model = prophet_model
        self.lstm_model = lstm_model
        
    def fit(self, df: pd.DataFrame):
        # Models are assumed to be pre-trained before being passed to the ensemble
        logging.info("Ensemble model ready. Models are pre-trained.")
        return self

    def predict(self, future_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
        """Combines predictions from Prophet and LSTM."""
        logging.info("Predicting with Ensemble model (Prophet + LSTM)...")
        # Get Prophet predictions
        prophet_forecast = self.prophet_model.predict(future_df)
        
        # Get LSTM predictions
        lstm_forecast = self.lstm_model.predict(future_df, historical_df)
        
        # Merge and average
        final_forecast = pd.merge(prophet_forecast, lstm_forecast, on='ds')
        final_forecast['yhat'] = (final_forecast['yhat'] + final_forecast['yhat_lstm']) / 2
        
        return final_forecast

def evaluate_model(model, validation_df: pd.DataFrame, model_type: str, historical_df: pd.DataFrame = None) -> float:
    """Evaluates a model's performance on a validation set."""
    if model_type == 'prophet':
        forecast = model.predict(validation_df[['ds', 'store_count', 'was_stocked_out']])
        merged = pd.merge(validation_df, forecast, on='ds')
        rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    elif model_type == 'lstm':
        # LSTM prediction requires historical context
        forecast = model.predict(validation_df, historical_df)
        merged = pd.merge(validation_df, forecast, on='ds')
        rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat_lstm']))
    else:
        raise ValueError("Unknown model type")
    
    logging.info(f"{model_type.upper()} Validation RMSE: {rmse:.4f}")
    return rmse