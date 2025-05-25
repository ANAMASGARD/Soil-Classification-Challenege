# Soil Image Classification Challenge

## Overview
This project is focused on the classification of soil types using image data and deep learning techniques. The goal is to accurately predict the type of soil from images, which is a common task in agricultural and environmental applications. The project is structured for participation in a Kaggle-style competition, with a workflow that includes data preprocessing, model training, evaluation, and submission generation.

## Project Structure
```
challenge-1/
├── requirements.txt         # All Python dependencies
├── data/                   # (User-provided) Directory for raw and processed data
├── docs/
│   └── cards/              # Documentation and metric cards
│       ├── ml-metrics.json # Example: F1 scores for each class
│       ├── Part-1.jpeg     # Project documentation images
│       └── Part-2.jpeg
├── notebooks/
│   └── main.py             # Main end-to-end pipeline script
├── src/
│   ├── preprocessing.py    # Data loading and preprocessing functions
│   ├── model_training.py   # Model building, training, and evaluation
│   └── postprocessing.py   # Test prediction and submission file generation
└── LICENSE                 # MIT License
```

## How It Works

### 1. Data Preparation
- The dataset consists of labeled soil images for training and unlabeled images for testing.
- Labels are provided in a CSV file mapping image IDs to soil types.
- Images are loaded, resized, and normalized for model input.

### 2. Preprocessing
- `src/preprocessing.py` handles reading the CSV, encoding labels, and converting images to arrays.

### 3. Model Training
- A Convolutional Neural Network (CNN) is built using TensorFlow/Keras (`src/model_training.py`).
- The model is trained with class balancing to address any class imbalance in the dataset.
- Training progress and evaluation metrics (accuracy, per-class F1 scores, confusion matrix) are displayed.

### 4. Evaluation
- The model is evaluated on a validation split.
- Key metrics include overall accuracy and the minimum per-class F1 score (as used in the competition).
- Visualizations such as confusion matrices are generated for analysis.

### 5. Submission
- `src/postprocessing.py` generates predictions for the test set and creates a submission CSV file in the required format.

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data:**
   - Place the training and test images, along with the label CSV, in the appropriate directories as expected by the scripts.

3. **Run the main pipeline:**
   - You can use the notebook (`notebooks/main.py`) or modular scripts in `src/` for each stage.
   - Example (from the root of `challenge-1/`):
     ```bash
     python notebooks/main.py
     ```

4. **Check outputs:**
   - Evaluation metrics and plots will be shown during training.
   - The final submission file will be saved as `submission.csv` in the output directory.

## Dependencies
All required Python packages are listed in `requirements.txt`, including:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tensorflow
- kaggle

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- Kaggle for the competition framework and dataset.
- TensorFlow and scikit-learn for machine learning tools.

---
For more details, see the code and documentation in the `docs/` folder.
