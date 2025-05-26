"""
Author: Annam.ai IIT Ropar
Team Name: SoilClassifiers
Team Members: Member-1, Member-2, Member-3, Member-4, Member-5
Leaderboard Rank: <Your Rank>
"""

import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def load_and_preprocess_image(image_path):
    """Load and preprocess an image for feature extraction."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((224, 224, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

def create_feature_extractor():
    """Create a feature extractor using EfficientNetB0 pre-trained on ImageNet."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def apply_data_augmentation(train_images):
    """Apply data augmentation to training images."""
    # Data augmentation configuration
    datagen = ImageDataGenerator(
        rotation_range=20,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        horizontal_flip=True
    )
    
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
    
    return all_train_images

def load_dataset_paths(base_path):
    """Load and prepare dataset paths."""
    train_path = base_path + 'train/'
    test_path = base_path + 'test/'
    train_labels_path = base_path + 'train_labels.csv'
    test_ids_path = base_path + 'test_ids.csv'
    
    # Load the datasets
    train_labels = pd.read_csv(train_labels_path)
    test_ids = pd.read_csv(test_ids_path)
    
    # Add full image paths to the dataframes
    train_labels['image_path'] = train_labels['image_id'].apply(lambda x: os.path.join(train_path, x))
    test_ids['image_path'] = test_ids['image_id'].apply(lambda x: os.path.join(test_path, x))
    
    return train_labels, test_ids

def extract_features(images, feature_extractor, batch_size=32):
    """Extract features from images using the feature extractor."""
    features = feature_extractor.predict(images, batch_size=batch_size, verbose=1)
    return features

def preprocessing(base_path='/kaggle/input/soil-classification-part-2/soil_competition-2025/'):
    """
    Main preprocessing function that handles:
    1. Loading and preprocessing images
    2. Data augmentation
    3. Feature extraction using EfficientNetB0
    4. Train-validation split
    """
    print("Starting preprocessing pipeline...")
    
    # Step 1: Load dataset paths and labels
    print("Loading dataset paths...")
    train_labels, test_ids = load_dataset_paths(base_path)
    
    # Step 2: Load and preprocess training images
    print("Loading and preprocessing training images...")
    train_images = np.array([load_and_preprocess_image(path) for path in train_labels['image_path']])
    
    # Step 3: Apply data augmentation
    print("Applying data augmentation...")
    all_train_images = apply_data_augmentation(train_images)
    
    # Step 4: Create feature extractor
    print("Creating feature extractor...")
    feature_extractor = create_feature_extractor()
    
    # Step 5: Extract features from training images
    print("Extracting training features...")
    all_train_features = extract_features(all_train_images, feature_extractor)
    
    # Step 6: Split into train and validation sets (80-20 split)
    print("Splitting into train and validation sets...")
    val_size = int(0.2 * len(all_train_features))
    train_features = all_train_features[:-val_size]
    val_features = all_train_features[-val_size:]
    
    # Step 7: Load and preprocess test images
    print("Loading and preprocessing test images...")
    test_images = np.array([load_and_preprocess_image(path) for path in test_ids['image_path']])
    
    # Step 8: Extract features from test images
    print("Extracting test features...")
    test_features = extract_features(test_images, feature_extractor)
    
    print("Preprocessing completed successfully!")
    
    return {
        'train_features': train_features,
        'val_features': val_features,
        'test_features': test_features,
        'feature_extractor': feature_extractor,
        'test_ids': test_ids
    }

if __name__ == "__main__":
    print("This is the file for preprocessing")
    # Uncomment the line below to run preprocessing
    preprocessing_results = preprocessing()