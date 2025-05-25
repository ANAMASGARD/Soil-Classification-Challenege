# preprocessing.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_data(input_path, img_size=128):
    train_csv = os.path.join(input_path, "train_labels.csv")
    img_dir = os.path.join(input_path, "train")
    df = pd.read_csv(train_csv)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['soil_type'])

    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['image_id'])
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(img_size, img_size))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(row['label'])

    return np.array(images), np.array(labels), le, df
