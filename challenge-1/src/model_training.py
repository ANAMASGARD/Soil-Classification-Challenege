# model_training.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(img_size, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, X, y, le, epochs=20):
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, class_weight=class_weight_dict)

    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("✅ Accuracy:", accuracy_score(y_val, y_pred_classes))
    report = classification_report(y_val, y_pred_classes, target_names=le.classes_, output_dict=True)
    f1_scores = [report[label]['f1-score'] for label in le.classes_]
    min_f1 = min(f1_scores)

    print("\n📊 Per-Class F1 Scores:")
    for cls, score in zip(le.classes_, f1_scores):
        print(f"{cls}: {score:.4f}")
    print(f"\n🏁 Final Evaluation Metric (Min F1): {min_f1:.4f}")

    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return model
