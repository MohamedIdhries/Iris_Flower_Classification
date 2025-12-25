import argparse
import os
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.utils import load_data

def main():
    parser = argparse.ArgumentParser(description='Train a RandomForest model on the Iris dataset')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV dataset (optional). If omitted, uses sklearn.datasets.load_iris')
    parser.add_argument('--model', type=str, default='models/rf_pipeline.joblib',
                        help='Path to save trained model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees in the RandomForest')
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, le = load_data(args.data, test_size=args.test_size, random_state=args.random_state)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'Test accuracy: {acc:.4f}')

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    # Save both pipeline and label encoder so we can decode predictions
    dump({'pipeline': pipeline, 'label_encoder': le}, args.model)
    print(f'Model saved to {args.model}')

if __name__ == '__main__':
    main()
