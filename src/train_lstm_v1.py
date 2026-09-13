import os
import random
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# File path definitions
X_PATH = os.path.join("data", "processed", "X_sequences.npy")
Y_PATH = os.path.join("data", "processed", "y_sequences.npy")
MODEL_OUTPUT_PATH = os.path.join("data", "processed", "lstm_v1_clean.pt")
SCALER_OUTPUT_PATH = os.path.join("data", "processed", "lstm_v1_scaler.pkl")


# =====================================================================
# STANDALONE LSTM MODEL DEFINITION
# =====================================================================

class StandaloneLSTMForecastModel(nn.Module):
    """
    PyTorch LSTM Classifier for Next-State Network Attack Forecasting (v1 Baseline).

    Architecture:
    - Input size: 15
    - Hidden size: 64
    - Layers: 1
    - Batch first: True
    - Takes final timestep -> Linear(64, 2)
    - Outputs 2 raw unnormalized logits (No softmax inside model)
    """

    def __init__(self, input_size: int = 15, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 2):
        super(StandaloneLSTMForecastModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape      : (batch_size, sequence_length=5, num_features=15)
        logits shape : (batch_size, num_classes=2)
        """
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]  # Extract final timestep
        logits = self.fc(last_step_out)
        return logits


# =====================================================================
# UTILITY HELPERS
# =====================================================================

def set_reproducibility_seeds(seed: int = 42):
    """Sets random seeds for Python, NumPy, PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chronological_train_val_test_split(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15
):
    """Splits sequential dataset strictly chronologically (No Shuffling)."""
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def apply_standard_scaling_only(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    """
    Fits StandardScaler ONLY on X_train.reshape(-1, 15).
    Transforms Train, Validation, and Test splits using that identical fitted scaler.
    Reshapes output back to (N, 5, 15).
    """
    n_train, seq_len, num_features = X_train.shape
    n_val = len(X_val)
    n_test = len(X_test)

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, num_features)
    scaler.fit(X_train_flat)

    X_train_scaled = scaler.transform(X_train_flat).reshape(n_train, seq_len, num_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(n_val, seq_len, num_features)
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(n_test, seq_len, num_features)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def calculate_false_positive_rate(cm: np.ndarray) -> float:
    """Calculates False Positive Rate (FPR) = FP / (FP + TN)."""
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return 0.0


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device):
    """Evaluates PyTorch model on a DataLoader using argmax predictions."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * len(targets)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    return np.array(all_targets), np.array(all_preds), avg_loss


# =====================================================================
# MAIN EXPERIMENT PIPELINE
# =====================================================================

def run_experiment_lstm_v1():
    """Execution entry point for LSTM v1 (StandardScaler Only)."""
    # 1. Reproducibility & Device Initialization
    set_reproducibility_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==================================================")
    print("      LSTM v1 — STANDALONE BASELINE EXPERIMENT    ")
    print("==================================================")
    print(f"Device being used: {device}")

    # 2. Data Loading & Inspection
    print("\nStep 1: Loading sequence datasets...")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Sequence datasets missing at {X_PATH} or {Y_PATH}")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    # 3. Chronological Split (70% Train / 15% Val / 15% Test)
    print("\nStep 2: Performing chronological split (70% Train / 15% Val / 15% Test)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = chronological_train_val_test_split(X, y)

    print(f"Train set size      : {len(X_train)} samples")
    print(f"Validation set size : {len(X_val)} samples")
    print(f"Test set size       : {len(X_test)} samples (UNTOUCHED UNTIL FINAL TEST)")

    # Print Training Class Counts
    count_class_0 = np.sum(y_train == 0)
    count_class_1 = np.sum(y_train == 1)
    print(f"\nTraining set class distribution -> Benign (0): {count_class_0}, Infiltration (1): {count_class_1}")

    # 4. Preprocessing: StandardScaler Only (Fitted strictly on Training set)
    print("\nStep 3: Fitting StandardScaler strictly on Training split...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = apply_standard_scaling_only(X_train, X_val, X_test)

    os.makedirs(os.path.dirname(SCALER_OUTPUT_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    print(f"Saved fitted StandardScaler to: {SCALER_OUTPUT_PATH}")

    # 5. Convert to PyTorch DataLoaders (shuffle=False)
    batch_size = 32
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6. Calculate Loss Class Weights derived ONLY from Training Labels
    weight_0 = len(y_train) / (2.0 * count_class_0)
    weight_1 = len(y_train) / (2.0 * count_class_1)
    class_weights_tensor = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
    print(f"Calculated Loss Class Weights -> Class 0: {weight_0:.4f}, Class 1: {weight_1:.4f}")

    # 7. Model Instantiation & Architecture Print
    model = StandaloneLSTMForecastModel(input_size=15, hidden_size=64, num_layers=1, num_classes=2).to(device)
    print("\nModel Architecture:")
    print(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 8. Training Loop with Early Stopping (Patience = 7, Selected by Validation F1)
    epochs = 50
    patience = 7
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    print("\nStep 4: Beginning Training (Max Epochs = 50, Early Stopping Patience = 7)...")
    print("-" * 85)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)

        train_loss = running_loss / len(train_loader.dataset)

        val_targets, val_preds, val_loss = evaluate_model(model, val_loader, device)
        val_acc = accuracy_score(val_targets, val_preds)
        val_prec = precision_score(val_targets, val_preds, zero_division=0)
        val_rec = recall_score(val_targets, val_preds, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)

        is_best = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
            is_best = True
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f} "
            f"{'<-- NEW BEST SAVED' if is_best else ''}"
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch:02d} epochs.")
            break

    print("-" * 85)
    print(f"Training Complete. Best Validation Checkpoint Saved at Epoch {best_epoch:02d} (Best Val F1: {best_val_f1:.4f})")

    # 9. Final Evaluation on Untouched Test Set
    print("\nStep 5: Evaluating best validation checkpoint on untouched Test Set...")
    best_model = StandaloneLSTMForecastModel(input_size=15, hidden_size=64, num_layers=1, num_classes=2).to(device)
    best_model.load_state_dict(torch.load(MODEL_OUTPUT_PATH, weights_only=True))

    test_targets, test_preds, _ = evaluate_model(best_model, test_loader, device)

    test_acc = accuracy_score(test_targets, test_preds)
    test_prec = precision_score(test_targets, test_preds, zero_division=0)
    test_rec = recall_score(test_targets, test_preds, zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    test_cm = confusion_matrix(test_targets, test_preds)
    test_fpr = calculate_false_positive_rate(test_cm)

    print("\n==================================================")
    print("         FINAL TEST EVALUATION RESULTS            ")
    print("==================================================")
    print(f"Best Validation Epoch : {best_epoch}")
    print(f"Best Validation F1    : {best_val_f1:.4f}")
    print(f"Test Accuracy         : {test_acc:.4f}")
    print(f"Precision (Class 1)   : {test_prec:.4f}")
    print(f"Recall (Class 1)      : {test_rec:.4f}")
    print(f"F1 Score (Class 1)    : {test_f1:.4f}")
    print(f"False Positive Rate   : {test_fpr:.4f}")

    print("\nConfusion Matrix:")
    print(test_cm)

    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=["Benign (0)", "Infiltration (1)"], zero_division=0))
    print("==================================================\n")


if __name__ == "__main__":
    run_experiment_lstm_v1()