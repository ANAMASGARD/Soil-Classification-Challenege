# postprocessing.py

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_and_submit(model, le, input_path, output_path, img_size=128):
    test_dir = os.path.join(input_path, "test")
    test_images = []
    test_ids = []

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = load_img(img_path, target_size=(img_size, img_size))
        img_array = img_to_array(img) / 255.0
        test_images.append(img_array)
        test_ids.append(img_name)

    test_X = np.array(test_images)
    test_preds = model.predict(test_X)
    test_pred_classes = np.argmax(test_preds, axis=1)
    test_pred_labels = le.inverse_transform(test_pred_classes)

    submission = pd.DataFrame({
        'image_id': test_ids,
        'soil_type': test_pred_labels
    })

    submission_path = os.path.join(output_path, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"📁 Submission saved to {submission_path}")
