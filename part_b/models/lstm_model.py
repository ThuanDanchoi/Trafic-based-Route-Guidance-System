import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
import os

def load_processed_data(data_dir='processed_data'):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist. Please check the path.")

    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'), allow_pickle=True)
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)
        sites_train = np.load(os.path.join(data_dir, 'sites_train.npy'), allow_pickle=True)
        sites_test = np.load(os.path.join(data_dir, 'sites_test.npy'), allow_pickle=True)

        with open(os.path.join(data_dir, 'site_scalers.pkl'), 'rb') as f:
            site_scalers = pickle.load(f)

        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'sites_train': sites_train,
            'sites_test': sites_test,
            'site_scalers': site_scalers
        }

        return data

    except Exception as e:
        raise


def build_lstm_model(input_shape, output_shape=1):
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(output_shape)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model


def train_lstm_model(X_train, y_train, X_val, y_val, input_shape, output_shape=1, epochs=100, batch_size=32):
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    model = build_lstm_model(input_shape, output_shape)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            os.path.join('saved_models', 'lstm_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def evaluate_lstm_model(model, X_test, y_test, scaler=None):
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    y_pred = model.predict(X_test)

    if scaler is not None:
        try:
            dummy = np.zeros((y_pred.shape[0], 96))
            dummy[:, 0:y_pred.shape[1]] = y_pred
            y_pred_inverse = scaler.inverse_transform(dummy)[:, 0:y_pred.shape[1]]

            dummy = np.zeros((y_test.shape[0], 96))
            dummy[:, 0:y_test.shape[1]] = y_test
            y_test_inverse = scaler.inverse_transform(dummy)[:, 0:y_test.shape[1]]
        except:
            y_pred_inverse = y_pred
            y_test_inverse = y_test
    else:
        y_pred_inverse = y_pred
        y_test_inverse = y_test

    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    r2 = r2_score(y_test_inverse.flatten(), y_pred_inverse.flatten())
    mape = np.mean(np.abs((y_test_inverse - y_pred_inverse) / (y_test_inverse + 1e-5))) * 100

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'y_pred': y_pred_inverse,
        'y_test': y_test_inverse
    }


# Main execution
data = load_processed_data()
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
sites_train = data['sites_train']
sites_test = data['sites_test']
site_scalers = data['site_scalers']

input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = y_train.shape[1]

X_train_lstm, X_val, y_train_lstm, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

model, history = train_lstm_model(
    X_train_lstm, y_train_lstm,
    X_val, y_val,
    input_shape,
    output_shape,
    epochs=10,
    batch_size=32
)


model_metrics = {}

first_site = list(site_scalers.keys())[0]
lstm_metrics = evaluate_lstm_model(model, X_test, y_test, site_scalers[first_site])
model_metrics['LSTM'] = lstm_metrics

unique_sites = np.unique(sites_test)
site_metrics = {}

for site in unique_sites[:3]:
    site_mask = sites_test == site
    if np.sum(site_mask) == 0:
        continue

    X_test_site = X_test[site_mask]
    y_test_site = y_test[site_mask]

    if len(X_test_site) < 10:
        continue

    scaler = site_scalers.get(site, site_scalers[first_site])
    site_metrics[site] = evaluate_lstm_model(model, X_test_site, y_test_site, scaler)