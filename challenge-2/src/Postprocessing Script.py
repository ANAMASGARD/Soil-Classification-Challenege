"""
Author: Annam.ai IIT Ropar
Team Name: SoilClassifiers
Team Members: Member-1, Member-2, Member-3, Member-4, Member-5
Leaderboard Rank: <Your Rank>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def create_autoencoder(input_dim):
    """Create a lightweight autoencoder for anomaly detection."""
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(train_features, epochs=30, batch_size=32):
    """Train the autoencoder for anomaly detection."""
    print("Training autoencoder...")
    autoencoder = create_autoencoder(train_features.shape[1])
    autoencoder.fit(
        train_features, train_features,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.2
    )
    return autoencoder

def compute_reconstruction_errors(autoencoder, features):
    """Compute reconstruction errors for given features."""
    reconstructed = autoencoder.predict(features)
    errors = np.mean(np.square(features - reconstructed), axis=1)
    return errors

def select_optimal_threshold(train_errors, val_errors):
    """
    Select optimal threshold for anomaly detection using pseudo-labeling.
    """
    # Create pseudo labels for validation set
    val_pseudo_labels = np.ones(len(val_errors))  # Start with all as soil (1)
    error_threshold_for_pseudo = np.percentile(val_errors, 90)  # Top 10% errors as pseudo non-soil
    val_pseudo_labels[val_errors > error_threshold_for_pseudo] = 0  # Label high errors as non-soil
    
    # Test multiple thresholds and pick the best based on F1-score
    thresholds = [np.percentile(train_errors, p) for p in [50, 75, 90]]
    best_threshold = None
    best_f1 = 0
    
    print("Selecting best threshold based on validation F1-score...")
    for threshold in thresholds:
        val_preds = np.where(val_errors <= threshold, 1, 0)
        f1 = f1_score(val_pseudo_labels, val_preds)
        print(f"Threshold: {threshold:.4f}, F1-score: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.4f} with F1-score: {best_f1:.4f}")
    return best_threshold, best_f1

def make_predictions(test_errors, threshold):
    """Make final predictions based on reconstruction errors and threshold."""
    test_preds = np.where(test_errors <= threshold, 1, 0)
    return test_preds

def create_submission_file(test_ids, predictions, filename='submission.csv'):
    """Create submission file with predictions."""
    submission = pd.DataFrame({
        'image_id': test_ids['image_id'],
        'label': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Submission file created: {filename}")
    return submission

def visualize_results(train_errors, test_errors, threshold):
    """Visualize reconstruction errors distribution."""
    plt.figure(figsize=(8, 6))
    sns.histplot(train_errors, label='Train Errors (Soil)', color='blue', bins=50, alpha=0.7)
    sns.histplot(test_errors, label='Test Errors', color='red', bins=50, alpha=0.7)
    plt.axvline(threshold, color='green', linestyle='--', label='Threshold')
    plt.title('Reconstruction Errors Distribution')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def print_prediction_statistics(predictions):
    """Print statistics about the predictions."""
    print(f"\nPrediction Statistics:")
    print(f"Soil predictions: {np.sum(predictions)} ({np.mean(predictions)*100:.1f}%)")
    print(f"Non-soil predictions: {len(predictions) - np.sum(predictions)} ({(1 - np.mean(predictions))*100:.1f}%)")

def postprocessing(train_features, val_features, test_features, test_ids, 
                  autoencoder_epochs=30, autoencoder_batch_size=32,
                  submission_filename='submission.csv', visualize=True):
    """
    Main postprocessing function that handles:
    1. Training autoencoder for anomaly detection
    2. Computing reconstruction errors
    3. Dynamic threshold selection
    4. Making final predictions
    5. Creating submission file
    6. Visualization and statistics
    """
    print("Starting postprocessing pipeline...")
    
    # Step 1: Train autoencoder
    autoencoder = train_autoencoder(train_features, autoencoder_epochs, autoencoder_batch_size)
    
    # Step 2: Compute reconstruction errors
    print("Computing reconstruction errors...")
    train_errors = compute_reconstruction_errors(autoencoder, train_features)
    val_errors = compute_reconstruction_errors(autoencoder, val_features)
    test_errors = compute_reconstruction_errors(autoencoder, test_features)
    
    # Step 3: Select optimal threshold
    best_threshold, best_f1 = select_optimal_threshold(train_errors, val_errors)
    
    # Step 4: Make final predictions
    print("Making final predictions...")
    test_predictions = make_predictions(test_errors, best_threshold)
    
    # Step 5: Create submission file
    submission = create_submission_file(test_ids, test_predictions, submission_filename)
    
    # Step 6: Visualize results (optional)
    if visualize:
        visualize_results(train_errors, test_errors, best_threshold)
    
    # Step 7: Print statistics
    print_prediction_statistics(test_predictions)
    
    print("Postprocessing completed successfully!")
    
    return {
        'autoencoder': autoencoder,
        'train_errors': train_errors,
        'val_errors': val_errors,
        'test_errors': test_errors,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'predictions': test_predictions,
        'submission': submission
    }

if __name__ == "__main__":
    print("This is the file for postprocessing")
    # Uncomment the lines below to run postprocessing with sample data
    # Note: You need to provide the actual features and test_ids from preprocessing
    postprocessing_results = postprocessing(train_features, val_features, test_features, test_ids)