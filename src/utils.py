import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

def load_data(path: Optional[str] = None, test_size: float = 0.2, random_state: int = 42
             ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load Iris data from CSV (if path provided) or from scikit-learn.

    Returns: X_train, X_test, y_train, y_test, label_encoder
    y values are integer encoded using LabelEncoder.
    """
    if path:
        df = pd.read_csv(path)
        # Expect either a 'species' column or last column is the target
        if 'species' in df.columns:
            X = df.drop(columns=['species'])
            y = df['species']
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
    else:
        data = load_iris(as_frame=True)
        # data.frame includes feature columns and a 'target' column
        X = data.frame.drop(columns=['target'])
        # Map numeric target to names to keep label encoder behavior consistent with CSV species names
        y = data.frame['target'].map(dict(enumerate(data.target_names)))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    return X_train, X_test, y_train, y_test, le
