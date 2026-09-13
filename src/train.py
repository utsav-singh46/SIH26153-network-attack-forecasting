import os
import joblib
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from model import flatten_sequences, build_logistic_regression_pipeline

# Data file paths
X_PATH = os.path.join("data", "processed", "X_sequences.npy")
Y_PATH = os.path.join("data", "processed", "y_sequences.npy")
MODEL_OUTPUT_PATH = os.path.join("data", "processed", "logistic_regression_baseline.pkl")


def chronological_train_val_test_split(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15
):
    """
    Chronological splitting is used because this is a temporal forecasting problem.
    Random splitting could leak information from later attack periods into training.

    Splits sequential dataset strictly by time:
    - Train: First 70%
    - Validation: Next 15%
    - Test: Final 15%
    """
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def calculate_false_positive_rate(cm: np.ndarray) -> float:
    """Calculates False Positive Rate (FPR) = FP / (FP + TN)."""
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return 0.0


def run_training_pipeline():
    """Main execution pipeline for restoring and evaluating the official scaled Logistic Regression baseline."""
    print("Step 1: Loading sequence datasets...")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError("Sequence data files missing. Please run src/features.py first.")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print(f"Original X shape : {X.shape}")
    print(f"Original y shape : {y.shape}")

    # Step 2: Flatten 3D sequence arrays to 2D
    print("\nStep 2: Flattening sequence dimensions (N, 5, 15) -> (N, 75)...")
    X_flat = flatten_sequences(X)

    # Step 3: Chronological Split (70% Train, 15% Val, 15% Test)
    print("\nStep 3: Performing chronological split (70% Train / 15% Val / 15% Test)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = chronological_train_val_test_split(X_flat, y)

    print(f"Train set size      : {len(X_train)} samples")
    print(f"Validation set size : {len(X_val)} samples")
    print(f"Test set size       : {len(X_test)} samples")

    # Step 4: Model Build & Training
    print("\nStep 4: Training official Scaled Logistic Regression baseline pipeline...")
    pipeline = build_logistic_regression_pipeline()

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)

    conv_warnings = [w for w in captured_warnings if issubclass(w.category, ConvergenceWarning)]
    print("\n--- Model Optimization Status ---")
    if conv_warnings:
        print("Convergence Status : FAILED TO CONVERGE")
    else:
        print("Convergence Status : SUCCESS (Model converged successfully)")

    # Step 5: Validation Check
    val_preds = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy : {val_acc:.4f}")

    # Step 6: Test Set Final Evaluation
    print("\nStep 6: Evaluating restored baseline pipeline on unseen Test Set...")
    test_preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)
    fpr = calculate_false_positive_rate(cm)

    print("\n--- Official Baseline Test Set Metrics ---")
    print(f"Test Accuracy        : {acc:.4f}")
    print(f"Precision (Class 1)  : {prec:.4f}")
    print(f"Recall (Class 1)     : {rec:.4f}")
    print(f"F1 Score (Class 1)   : {f1:.4f}")
    print(f"False Positive Rate  : {fpr:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, test_preds, target_names=["Benign (0)", "Infiltration (1)"], zero_division=0
        )
    )

    # Step 7: Export restored baseline artifact
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    model_payload = {
        "pipeline": pipeline,
        "input_shape": X.shape[1:],  # (5, 15)
        "flattened_features": X_flat.shape[1],  # 75
        "target_labels": {0: "Benign", 1: "Infiltration"},
        "model_type": "Official Scaled Logistic Regression Baseline Pipeline",
    }
    joblib.dump(model_payload, MODEL_OUTPUT_PATH)
    print(f"Saved official baseline pipeline artifact to: {MODEL_OUTPUT_PATH}\n")


if __name__ == "__main__":
    run_training_pipeline()