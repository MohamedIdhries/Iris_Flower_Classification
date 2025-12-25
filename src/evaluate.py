import argparse
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.utils import load_data

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the Iris dataset')
    parser.add_argument('--model', type=str, default='models/rf_pipeline.joblib', help='Path to saved model')
    parser.add_argument('--data', type=str, default=None, help='Optional CSV to evaluate on (otherwise uses sklearn iris)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction used if loading data')
    args = parser.parse_args()

    model_bundle = load(args.model)
    pipeline = model_bundle['pipeline']
    le = model_bundle['label_encoder']

    _, X_test, _, y_test, _ = load_data(args.data, test_size=args.test_size)

    preds = pipeline.predict(X_test)
    try:
        probs = pipeline.predict_proba(X_test)
    except Exception:
        probs = None

    print('Accuracy:', accuracy_score(y_test, preds))
    print('\nClassification report:')
    print(classification_report(y_test, preds, target_names=le.classes_))
    print('\nConfusion matrix:')
    print(confusion_matrix(y_test, preds))

if __name__ == '__main__':
    main()
