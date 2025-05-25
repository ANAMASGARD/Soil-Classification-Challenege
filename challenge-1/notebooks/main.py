
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === Step 1: Set up paths ===
# Kaggle competition dataset location
input_path = '/kaggle/input/soil-classification/soil_classification-2025'
output_path = '/kaggle/working/'

# Verify paths
print("Input directory contents:")
!ls {input_path}

# === Step 2: Load labels ===
train_csv = os.path.join(input_path, "train_labels.csv")
img_dir = os.path.join(input_path, "train")
df = pd.read_csv(train_csv)

# === Step 3: Label encoding ===
le = LabelEncoder()
df['label'] = le.fit_transform(df['soil_type'])

# === Step 4: Load and preprocess images ===
IMG_SIZE = 128
images = []
labels = []

for _, row in df.iterrows():
    img_path = os.path.join(img_dir, row['image_id'])
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(row['label'])

X = np.array(images)
y = np.array(labels)

# === Step 5: Train/Val split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Step 6: Handle class imbalance ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# === Step 7: Model architecture ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Step 8: Training ===
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, class_weight=class_weight_dict)

# === Step 9: Evaluation ===
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

print("✅ Accuracy:", accuracy_score(y_val, y_pred_classes))

# Classification report
report = classification_report(y_val, y_pred_classes, target_names=le.classes_, output_dict=True)
f1_scores = [report[label]['f1-score'] for label in le.classes_]
min_f1 = min(f1_scores)

print("\n📊 Per-Class F1 Scores:")
for cls, score in zip(le.classes_, f1_scores):
    print(f"{cls}: {score:.4f}")
print(f"\n🏁 Final Evaluation Metric (Min F1): {min_f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# === Step 10: Generate predictions for test set (for competition submission) ===
# Load test images
test_dir = os.path.join(input_path, "test")
test_images = []
test_ids = []

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    test_images.append(img_array)
    test_ids.append(img_name)

test_X = np.array(test_images)

# Predict
test_preds = model.predict(test_X)
test_pred_classes = np.argmax(test_preds, axis=1)
test_pred_labels = le.inverse_transform(test_pred_classes)

# Create submission file
submission = pd.DataFrame({
    'image_id': test_ids,
    'soil_type': test_pred_labels
})

submission_path = os.path.join(output_path, 'submission.csv')
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")


 

