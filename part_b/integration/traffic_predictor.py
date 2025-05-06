"""
Traffic volume predictor using pre-trained deep learning models (LSTM, GRU, BiLSTM).
"""

import numpy as np
import os
import datetime
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.losses import MeanSquaredError

class TrafficPredictor:
    """
    Traffic prediction model wrapper.

    Loads a trained deep learning model (LSTM/GRU/BiLSTM) and associated scalers
    to make traffic volume predictions based on temporal features.
    """

    def __init__(self, model_type='lstm'):
        """
        Initialize and load the model and scaler.

        Args:
            model_type (str): Type of model to use ('lstm', 'gru', 'bilstm')
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_filename = f'{model_type}_model.h5'
        self.model_path = os.path.join(base_dir, '..', 'models', 'saved_models', model_filename)

        # Load model
        self.model = load_model(
            self.model_path,
            custom_objects={"mse": MeanSquaredError()}
        )

        # Load site-specific scalers (must have been saved during training)
        scaler_path = os.path.join(base_dir, '..', 'models', 'processed_data', 'site_scalers.pkl')
        self.scalers = joblib.load(scaler_path)

    def predict_for_time(self, datetime_obj, site_id, model_type='lstm'):
        """
        Predict traffic volume (vehicles per hour) for a given time and site.

        Args:
            datetime_obj (datetime.datetime): Time to make the prediction for
            site_id (int or str): SCATS site ID
            model_type (str): Type of model used ('lstm', 'gru', 'bilstm')

        Returns:
            float: Predicted traffic volume (vehicles/hour)
        """
        # Extract time-based features
        hour = datetime_obj.hour
        minute = datetime_obj.minute
        day_of_week = datetime_obj.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak = 1 if (7 <= hour <= 9 or 16 <= hour <= 18) and not is_weekend else 0

        input_vector = np.array([[hour, minute, day_of_week, is_weekend, is_peak, 0, 0, 0, 0]], dtype=np.float32)

        # Apply scaling if a scaler exists for the site
        if str(site_id) in self.scalers:
            scaler = self.scalers[str(site_id)]
            input_vector = scaler.transform(input_vector)

        # Reshape input for model: (batch_size, time_steps, features)
        input_vector = np.expand_dims(input_vector, axis=0)

        # Predict
        predicted_volume = self.model.predict(input_vector)
        return float(predicted_volume[0][0])
