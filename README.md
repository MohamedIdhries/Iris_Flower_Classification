# Iris Flower Classification

Iris Flower Classification is a small machine learning project that predicts the species of an iris flower (Setosa, Versicolor, Virginica) using sepal and petal measurements. This repository demonstrates end-to-end steps: data loading, preprocessing, model training, evaluation, and single-sample inference using scikit-learn.

## What's included
- Training, evaluation, and prediction scripts (src/train.py, src/evaluate.py, src/predict.py)
- Data helpers (src/utils.py)
- Requirements (requirements.txt)
- Simple RandomForest baseline pipeline saved/loaded with joblib

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- joblib

Install:
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

Train a model (uses scikit-learn's Iris dataset by default):
```
python src/train.py
```

Train using a CSV file (CSV must either have a `species` column or have the target as the last column):
```
python src/train.py --data data/iris.csv --model models/rf_pipeline.joblib
```

Evaluate a saved model:
```
python src/evaluate.py --model models/rf_pipeline.joblib
```

Predict a single sample:
```
python src/predict.py --model models/rf_pipeline.joblib \
  --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

## Scripts overview
- src/utils.py: data loading and preprocessing helper. load_data(path=None) will load sklearn's iris if path is omitted. Returns X_train, X_test, y_train, y_test, label_encoder.
- src/train.py: trains a pipeline (StandardScaler + RandomForest), prints test accuracy and saves a joblib bundle containing the pipeline and label encoder.
- src/evaluate.py: loads the saved bundle and prints accuracy, classification report, and confusion matrix on the test set.
- src/predict.py: loads the saved bundle and predicts a single sample from command-line feature values.

## Model artifact
The scripts save a joblib bundle (default: `models/rf_pipeline.joblib`) that contains:
```
{ 'pipeline': sklearn_pipeline, 'label_encoder': sklearn.preprocessing.LabelEncoder }
```
This makes decoding integer predictions back to species names straightforward.

## Project structure
- data/                 # raw and processed datasets (e.g., data/iris.csv)
- src/                  # training/eval/predict scripts and helpers
- models/               # saved model artifacts (e.g., models/rf_pipeline.joblib)
- requirements.txt
- README.md

## Notes & next steps
- Add an example `data/iris.csv` if you want to train on a CSV rather than the built-in dataset.
- Consider adding unit tests and a CI workflow (GitHub Actions) to run linting and test training on push.
- License: add a LICENSE file (e.g., MIT) if you want to set the repository license.

## Author
Mohamed Idhries
