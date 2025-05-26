"""
Soil Classification using Autoencoder-based Anomaly Detection
Author: Annam.ai IIT Ropar
Team Name: SoilClassifiers
Team Members: Member-1, Member-2, Member-3, Member-4, Member-5
Leaderboard Rank: <Your Rank>
"""

# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.metrics import f1_score

# Import custom preprocessing and postprocessing modules
from preprocessing import preprocessing
from postprocessing import postprocessing

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    """
    Main function that orchestrates the entire soil classification pipeline.
    """
    print("=" * 60)
    print("SOIL CLASSIFICATION USING AUTOENCODER-BASED ANOMALY DETECTION")
    print("=" * 60)
    
    # Define paths to the dataset
    BASE_PATH = '/kaggle/input/soil-classification-part-2/soil_competition-2025/'
    
    # Phase 1: Preprocessing
    print("\n🔄 Phase 1: Preprocessing")
    print("-" * 30)
    
    preprocessing_results = preprocessing(BASE_PATH)
    
    # Extract results from preprocessing
    train_features = preprocessing_results['train_features']
    val_features = preprocessing_results['val_features']
    test_features = preprocessing_results['test_features']
    feature_extractor = preprocessing_results['feature_extractor']
    test_ids = preprocessing_results['test_ids']
    
    print(f"✅ Training features shape: {train_features.shape}")
    print(f"✅ Validation features shape: {val_features.shape}")
    print(f"✅ Test features shape: {test_features.shape}")
    
    # Phase 2: Postprocessing
    print("\n🔄 Phase 2: Postprocessing")
    print("-" * 30)
    
    postprocessing_results = postprocessing(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        test_ids=test_ids,
        autoencoder_epochs=30,
        autoencoder_batch_size=32,
        submission_filename='submission.csv',
        visualize=True
    )
    
    # Extract final results
    best_threshold = postprocessing_results['best_threshold']
    best_f1 = postprocessing_results['best_f1']
    predictions = postprocessing_results['predictions']
    
    print(f"✅ Best threshold: {best_threshold:.4f}")
    print(f"✅ Best F1-score: {best_f1:.4f}")
    print(f"✅ Total predictions: {len(predictions)}")
    
    # Final Summary
    print("\n🎯 Final Summary")
    print("-" * 30)
    print(f"📊 Model Performance:")
    print(f"   • Threshold: {best_threshold:.4f}")
    print(f"   • Validation F1-score: {best_f1:.4f}")
    print(f"   • Soil predictions: {np.sum(predictions)} ({np.mean(predictions)*100:.1f}%)")
    print(f"   • Non-soil predictions: {len(predictions) - np.sum(predictions)} ({(1 - np.mean(predictions))*100:.1f}%)")
    print(f"📁 Output: submission.csv created successfully!")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)
    
    return postprocessing_results

# Alternative approach: Run the original code directly (for compatibility)
def run_original_pipeline():
    """
    Run the original pipeline as provided in the uploaded code.
    This function maintains the exact structure of the original code.
    """
    # Define paths to the dataset
    BASE_PATH = '/kaggle/input/soil-classification-part-2/soil_competition-2025/'
    TRAIN_PATH = BASE_PATH + 'train/'
    TEST_PATH = BASE_PATH + 'test/'
    TRAIN_LABELS = BASE_PATH + 'train_labels.csv'
    TEST_IDS = BASE_PATH + 'test_ids.csv'

    # Load the datasets
    train_labels = pd.read_csv(TRAIN_LABELS)
    test_ids = pd.read_csv(TEST_IDS)

    # Add full image paths to the dataframes
    train_labels['image_path'] = train_labels['image_id'].apply(lambda x: os.path.join(TRAIN_PATH, x))
    test_ids['image_path'] = test_ids['image_id'].apply(lambda x: os.path.join(TEST_PATH, x))

    # Step 1: Feature Extraction using Pre-trained EfficientNetB0
    def create_feature_extractor():
        """Create a feature extractor using EfficientNetB0 pre-trained on ImageNet."""
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        model = Model(inputs=base_model.input, outputs=x)
        return model

    def load_and_preprocess_image(image_path):
        """Load and preprocess an image for feature extraction."""
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros((224, 224, 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        return img

    # Data augmentation for training images
    datagen = ImageDataGenerator(
        rotation_range=20,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Load and preprocess images
    print("Loading training images...")
    train_images = np.array([load_and_preprocess_image(path) for path in train_labels['image_path']])

    # Apply augmentation to training images
    augmented_images = []
    for img in train_images:
        img = img.reshape((1,) + img.shape)  # Reshape for ImageDataGenerator
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0])
            break  # Take only one augmented image per original
    augmented_images = np.array(augmented_images)

    # Combine original and augmented images
    all_train_images = np.concatenate([train_images, augmented_images], axis=0)

    # Extract features
    print("Extracting training features...")
    feature_extractor = create_feature_extractor()
    all_train_features = feature_extractor.predict(all_train_images, batch_size=32, verbose=1)

    # Split into train and validation sets (80-20 split)
    val_size = int(0.2 * len(all_train_features))
    train_features = all_train_features[:-val_size]
    val_features = all_train_features[-val_size:]

    # Extract test features
    print("Extracting test features...")
    test_images = np.array([load_and_preprocess_image(path) for path in test_ids['image_path']])
    test_features = feature_extractor.predict(test_images, batch_size=32, verbose=1)

    # Step 2: Autoencoder for Anomaly Detection
    def create_autoencoder(input_dim):
        """Create a lightweight autoencoder for anomaly detection."""
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='linear')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    # Train the autoencoder
    print("Training autoencoder...")
    autoencoder = create_autoencoder(train_features.shape[1])
    autoencoder.fit(
        train_features, train_features,
        epochs=30,  # Reduced epochs for faster training
        batch_size=32,
        verbose=1,
        validation_split=0.2
    )

    # Compute reconstruction errors
    print("Computing reconstruction errors...")
    train_recon = autoencoder.predict(train_features)
    train_errors = np.mean(np.square(train_features - train_recon), axis=1)

    val_recon = autoencoder.predict(val_features)
    val_errors = np.mean(np.square(val_features - val_recon), axis=1)

    test_recon = autoencoder.predict(test_features)
    test_errors = np.mean(np.square(test_features - test_recon), axis=1)

    # Step 3: Dynamic Threshold Selection
    # Simulate non-soil images in validation set by taking high-error samples
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

    # Apply the best threshold to test predictions
    test_preds = np.where(test_errors <= best_threshold, 1, 0)

    # Step 4: Create Submission File
    submission = pd.DataFrame({
        'image_id': test_ids['image_id'],
        'label': test_preds
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

    # Step 5: Visualize Reconstruction Errors (for Judges)
    plt.figure(figsize=(8, 6))
    sns.histplot(train_errors, label='Train Errors (Soil)', color='blue', bins=50, alpha=0.7)
    sns.histplot(test_errors, label='Test Errors', color='red', bins=50, alpha=0.7)
    plt.axvline(best_threshold, color='green', linestyle='--', label='Threshold')
    plt.title('Reconstruction Errors Distribution')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Print prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"Soil predictions: {np.sum(test_preds)} ({np.mean(test_preds)*100:.1f}%)")
    print(f"Non-soil predictions: {len(test_preds) - np.sum(test_preds)} ({(1 - np.mean(test_preds))*100:.1f}%)")

if __name__ == "__main__":
    # Choose which pipeline to run:
    
    # Option 1: Modularized pipeline (recommended)
    print("Running modularized pipeline...")
    results = main()
    
    # Option 2: Original pipeline (uncomment to use)
    print("Running original pipeline...")
    run_original_pipeline()