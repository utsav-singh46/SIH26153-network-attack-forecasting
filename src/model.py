import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Flattens 3D sequence arrays into a 2D matrix for traditional ML classifiers.

    Input shape  : (N, 5, 15) -> (N samples, 5 time steps, 15 features)
    Output shape : (N, 75)   -> (N samples, 75 flattened features)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D input array (samples, time_steps, features), got shape {X.shape}")

    num_samples, seq_len, num_features = X.shape
    flattened_dim = seq_len * num_features

    # Reshape while maintaining strict temporal ordering
    X_flat = X.reshape(num_samples, flattened_dim)
    return X_flat


def build_logistic_regression_pipeline(random_state: int = 42) -> Pipeline:
    """
    Constructs the official baseline Scaled Logistic Regression Pipeline.

    - StandardScaler ensures feature scaling so optimization converges cleanly.
    - LogisticRegression uses default C=1.0 and class_weight='balanced' with max_iter=5000.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return pipeline