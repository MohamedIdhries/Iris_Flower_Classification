import argparse
from joblib import load
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Predict species for a single iris sample')
    parser.add_argument('--model', type=str, default='models/rf_pipeline.joblib', help='Path to saved model')
    parser.add_argument('--sepal-length', type=float, required=True)
    parser.add_argument('--sepal-width', type=float, required=True)
    parser.add_argument('--petal-length', type=float, required=True)
    parser.add_argument('--petal-width', type=float, required=True)
    args = parser.parse_args()

    model_bundle = load(args.model)
    pipeline = model_bundle['pipeline']
    le = model_bundle['label_encoder']

    features = np.array([[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]])
    pred = pipeline.predict(features)[0]
    try:
        prob = pipeline.predict_proba(features)[0]
    except Exception:
        prob = None

    species = le.inverse_transform([int(pred)])[0]
    if prob is not None:
        print(f'Predicted species: {species} (probabilities={prob})')
    else:
        print(f'Predicted species: {species}')

if __name__ == '__main__':
    main()
