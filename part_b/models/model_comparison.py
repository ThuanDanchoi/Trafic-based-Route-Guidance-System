import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

def load_processed_data(data_dir='processed_data'):
    """Load preprocessed data for model evaluation"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist. Please check the path.")

    try:
        # Load test data only since we're just evaluating
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)
        sites_test = np.load(os.path.join(data_dir, 'sites_test.npy'), allow_pickle=True)

        with open(os.path.join(data_dir, 'site_scalers.pkl'), 'rb') as f:
            site_scalers = pickle.load(f)

        # Convert to float32 to prevent dtype issues
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

        return X_test, y_test, sites_test, site_scalers

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def load_models(models_dir='saved_models'):
    """Load all trained models"""
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Directory {models_dir} does not exist. Please check the path.")

    models = {}
    try:
        # Load LSTM model
        lstm_path = os.path.join(models_dir, 'lstm_model.h5')
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError()
        }
        if os.path.exists(lstm_path):
            models['LSTM'] = load_model(lstm_path, custom_objects=custom_objects)
            print(f"Loaded LSTM model from {lstm_path}")
        else:
            print(f"Warning: LSTM model not found at {lstm_path}")

        # Load GRU model
        gru_path = os.path.join(models_dir, 'gru_model.h5')
        if os.path.exists(gru_path):
            models['GRU'] = load_model(gru_path, custom_objects=custom_objects)
            print(f"Loaded GRU model from {gru_path}")
        else:
            print(f"Warning: GRU model not found at {gru_path}")

        # Load BiLSTM model
        bilstm_path = os.path.join(models_dir, 'bilstm_model.h5')
        if os.path.exists(bilstm_path):
            models['BiLSTM'] = load_model(bilstm_path, custom_objects=custom_objects)
            print(f"Loaded BiLSTM model from {bilstm_path}")
        else:
            print(f"Warning: BiLSTM model not found at {bilstm_path}")

        return models

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise


def evaluate_model(model, X_test, y_test, scaler=None):
    """Evaluate a single model and return metrics"""
    y_pred = model.predict(X_test)

    # Inverse transform predictions and actual values if scaler is provided
    if scaler is not None:
        try:
            dummy = np.zeros((y_pred.shape[0], 96))
            dummy[:, 0:y_pred.shape[1]] = y_pred
            y_pred_inverse = scaler.inverse_transform(dummy)[:, 0:y_pred.shape[1]]

            dummy = np.zeros((y_test.shape[0], 96))
            dummy[:, 0:y_test.shape[1]] = y_test
            y_test_inverse = scaler.inverse_transform(dummy)[:, 0:y_test.shape[1]]
        except:
            print("Warning: Could not inverse transform. Using scaled values.")
            y_pred_inverse = y_pred
            y_test_inverse = y_test
    else:
        y_pred_inverse = y_pred
        y_test_inverse = y_test

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    r2 = r2_score(y_test_inverse.flatten(), y_pred_inverse.flatten())
    mape = np.mean(np.abs((y_test_inverse - y_pred_inverse) / (y_test_inverse + 1e-5))) * 100

    # Return metrics as dictionary
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'y_pred': y_pred_inverse,
        'y_test': y_test_inverse
    }


def evaluate_all_models(models, X_test, y_test, scaler):
    """Evaluate all models and return comparative metrics"""
    results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} model...")
        results[model_name] = evaluate_model(model, X_test, y_test, scaler)

        # Print metrics
        print(f"  RMSE: {results[model_name]['rmse']:.4f}")
        print(f"  MAE: {results[model_name]['mae']:.4f}")
        print(f"  R²: {results[model_name]['r2']:.4f}")
        print(f"  MAPE: {results[model_name]['mape']:.2f}%")

    return results


def plot_comparison(results):
    """Create visualizations comparing model performance"""
    model_names = list(results.keys())

    # Create directory for plots if it doesn't exist
    os.makedirs('comparison_plots', exist_ok=True)

    # Plot metrics comparison
    metrics = ['rmse', 'mae', 'mape']
    metric_names = ['Root Mean Squared Error', 'Mean Absolute Error', 'Mean Absolute Percentage Error (%)']

    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        values = [results[model][metric] for model in model_names]
        bars = plt.bar(model_names, values)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.title(metric_names[i])
        plt.ylabel('Value (lower is better)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot R² comparison (higher is better)
    plt.subplot(2, 2, 4)
    r2_values = [results[model]['r2'] for model in model_names]
    bars = plt.bar(model_names, r2_values)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.title('R² Score')
    plt.ylabel('Value (higher is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('comparison_plots/metrics_comparison.png')
    print("Saved metrics comparison plot to 'comparison_plots/metrics_comparison.png'")

    # Plot predictions comparison
    plt.figure(figsize=(15, 8))

    # Get a sample of the test data (first 100 points)
    sample_size = min(100, len(results[model_names[0]]['y_test']))

    # Plot actual values
    plt.plot(results[model_names[0]]['y_test'][:sample_size, 0], 'k-',
             linewidth=2, label='Actual')

    # Plot predicted values for each model
    colors = ['b', 'r', 'g', 'c', 'm']
    for i, model in enumerate(model_names):
        plt.plot(results[model]['y_pred'][:sample_size, 0],
                 f'{colors[i % len(colors)]}--',
                 linewidth=1.5, label=f'{model} Prediction')

    plt.title('Model Predictions Comparison')
    plt.xlabel('Sample')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_plots/predictions_comparison.png')
    print("Saved predictions comparison plot to 'comparison_plots/predictions_comparison.png'")


def find_best_model(results):
    """Determine the best model based on metrics"""
    # Create a scoring system based on metrics
    # Lower values are better for RMSE, MAE, MAPE
    # Higher values are better for R²

    # Normalize each metric to 0-1 range for fair comparison
    model_names = list(results.keys())
    metrics = {}

    # Gather metrics
    for metric in ['rmse', 'mae', 'mape']:
        values = np.array([results[model][metric] for model in model_names])
        min_val, max_val = values.min(), values.max()
        if max_val > min_val:  # Avoid division by zero
            # Normalize and invert (lower is better for these metrics)
            metrics[metric] = 1 - (values - min_val) / (max_val - min_val)
        else:
            metrics[metric] = np.ones_like(values)

    # For R², higher is better
    r2_values = np.array([results[model]['r2'] for model in model_names])
    min_r2, max_r2 = r2_values.min(), r2_values.max()
    if max_r2 > min_r2:
        metrics['r2'] = (r2_values - min_r2) / (max_r2 - min_r2)
    else:
        metrics['r2'] = np.ones_like(r2_values)

    # Calculate overall score (equal weight for each metric)
    scores = np.zeros(len(model_names))
    for metric in metrics:
        scores += metrics[metric]

    # Find best model
    best_idx = np.argmax(scores)
    best_model = model_names[best_idx]

    print("\nModel Ranking:")
    for i, model in enumerate(model_names):
        idx = np.argsort(scores)[::-1][i]  # Sort by descending score
        print(f"{i + 1}. {model_names[idx]} (Score: {scores[idx]:.4f})")

    print(f"\nBest model: {best_model}")
    print(f"RMSE: {results[best_model]['rmse']:.4f}")
    print(f"MAE: {results[best_model]['mae']:.4f}")
    print(f"R²: {results[best_model]['r2']:.4f}")
    print(f"MAPE: {results[best_model]['mape']:.2f}%")

    return best_model


def export_results(results, best_model):
    """Export results to CSV for future reference"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Prepare data for CSV
    data = []
    model_names = list(results.keys())

    for model in model_names:
        data.append({
            'Model': model,
            'RMSE': results[model]['rmse'],
            'MAE': results[model]['mae'],
            'R²': results[model]['r2'],
            'MAPE': results[model]['mape'],
            'Best Model': model == best_model
        })

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    csv_path = 'results/model_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    return df


# Main execution
if __name__ == "__main__":
    print("Starting model comparison...")

    # Load test data
    X_test, y_test, sites_test, site_scalers = load_processed_data()
    print(f"Loaded test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

    # Load trained models
    models = load_models()

    if not models:
        print("No models found. Please check that the models were saved correctly.")
        exit(1)

    # Get first site scaler for evaluation
    first_site = list(site_scalers.keys())[0]
    scaler = site_scalers[first_site]

    # Evaluate all models
    results = evaluate_all_models(models, X_test, y_test, scaler)

    # Plot comparison
    plot_comparison(results)

    # Find best model
    best_model = find_best_model(results)

    # Export results
    export_results(results, best_model)

    print("\nModel comparison complete!")