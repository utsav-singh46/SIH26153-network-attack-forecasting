import os
import joblib
import numpy as np
import torch
import torch.nn as nn

# =====================================================================
# DEFAULT PATH DEFINITIONS
# =====================================================================
# Uses the trained v1 model checkpoint and its fitted scaler by default
DEFAULT_CHECKPOINT_PATH = os.path.join("data", "processed", "lstm_v1_clean.pt")
DEFAULT_SCALER_PATH = os.path.join("data", "processed", "lstm_v1_scaler.pkl")

# Human-readable class name mapping
CLASS_MAP = {
    0: "Benign",
    1: "Infiltration"
}


# =====================================================================
# MODEL ARCHITECTURE DEFINITION
# =====================================================================
class InferenceLSTMModel(nn.Module):
    """
    Standard PyTorch LSTM Classifier matching the v1 architecture.
    Input : (batch_size, sequence_length=5, num_features=15)
    Output: 2 raw unnormalized logits
    """
    def __init__(self, input_size: int = 15, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 2):
        super(InferenceLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]  # Take final timestep output
        logits = self.fc(last_step_out)
        return logits


# =====================================================================
# HELPER FUNCTIONS: LOADING MODEL AND SCALER
# =====================================================================
def load_trained_model(checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, device: str = "cpu") -> nn.Module:
    """
    Loads the PyTorch LSTM model weights from disk and sets it to evaluation mode.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    model = InferenceLSTMModel(input_size=15, hidden_size=64, num_layers=1, num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set model to evaluation mode (disables dropout/batchnorm training behavior)
    return model


def load_scaler(scaler_path: str = DEFAULT_SCALER_PATH):
    """
    Loads the fitted StandardScaler from disk.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler pickle file not found at: {scaler_path}")
    
    return joblib.load(scaler_path)


# =====================================================================
# CORE PREDICTION FUNCTIONS
# =====================================================================
def predict_sequence(
    sequence: np.ndarray,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH
) -> dict:
    """
    Runs inference on a SINGLE sequence input.

    Parameters:
        sequence (np.ndarray): Input array of shape (5, 15) or (1, 5, 15).
        checkpoint_path (str): Path to the trained model checkpoint (.pt).
        scaler_path (str)    : Path to the fitted StandardScaler (.pkl).

    Returns:
        dict: A dictionary containing:
            - predicted_class (int)
            - class_name (str)
            - infiltration_probability (float)
            - benign_probability (float)
    """
    # 1. Input Shape Validation & Formatting
    seq = np.array(sequence, dtype=np.float32)
    if seq.ndim == 2:
        if seq.shape != (5, 15):
            raise ValueError(f"Expected single sequence of shape (5, 15), got {seq.shape}")
        seq = np.expand_dims(seq, axis=0)  # Shape becomes (1, 5, 15)
    elif seq.ndim == 3:
        if seq.shape[1:] != (5, 15):
            raise ValueError(f"Expected sequence dimensions (N, 5, 15), got {seq.shape}")
    else:
        raise ValueError(f"Invalid input dimensions: {seq.ndim}D array provided.")

    # 2. Load Preprocessing Scaler & Transform Input
    scaler = load_scaler(scaler_path)
    # Flatten sequence to 2D -> (5, 15), scale features, then reshape back to (1, 5, 15)
    seq_flat = seq.reshape(-1, 15)
    seq_scaled_flat = scaler.transform(seq_flat)
    seq_scaled = seq_scaled_flat.reshape(1, 5, 15)

    # 3. Load Model & Convert Input to PyTorch Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(checkpoint_path, device=device.type)
    tensor_input = torch.tensor(seq_scaled, dtype=torch.float32).to(device)

    # 4. Perform Inference Pass
    with torch.no_grad():
        logits = model(tensor_input)
        # Convert raw logits to probabilities using Softmax
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(torch.argmax(logits, dim=1).cpu().numpy()[0])

    benign_prob = float(probabilities[0])
    infiltration_prob = float(probabilities[1])

    result = {
        "predicted_class": predicted_class,
        "class_name": CLASS_MAP[predicted_class],
        "benign_probability": round(benign_prob, 4),
        "infiltration_probability": round(infiltration_prob, 4),
    }

    return result


def predict_batch(
    sequences: np.ndarray,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH
) -> list:
    """
    Runs inference on MULTIPLE sequence inputs in batch mode.

    Parameters:
        sequences (np.ndarray): Array of shape (N, 5, 15).

    Returns:
        list of dicts: List of prediction result dictionaries.
    """
    seqs = np.array(sequences, dtype=np.float32)
    if seqs.ndim != 3 or seqs.shape[1:] != (5, 15):
        raise ValueError(f"Expected batch shape (N, 5, 15), got {seqs.shape}")

    n_samples = seqs.shape[0]

    # Preprocess entire batch
    scaler = load_scaler(scaler_path)
    seqs_flat = seqs.reshape(-1, 15)
    seqs_scaled_flat = scaler.transform(seqs_flat)
    seqs_scaled = seqs_scaled_flat.reshape(n_samples, 5, 15)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(checkpoint_path, device=device.type)
    tensor_inputs = torch.tensor(seqs_scaled, dtype=torch.float32).to(device)

    # Batch Inference
    with torch.no_grad():
        logits = model(tensor_inputs)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()

    results = []
    for i in range(n_samples):
        pred_cls = int(predicted_classes[i])
        ben_prob = float(probabilities[i][0])
        inf_prob = float(probabilities[i][1])

        results.append({
            "sample_index": i,
            "predicted_class": pred_cls,
            "class_name": CLASS_MAP[pred_cls],
            "benign_probability": round(ben_prob, 4),
            "infiltration_probability": round(inf_prob, 4),
        })

    return results


# =====================================================================
# DEMO EXECUTION / VERIFICATION SCRIPT
# =====================================================================
if __name__ == "__main__":
    print("==================================================")
    print("      SIH26153 PROTOTYPE PREDICTION PIPELINE       ")
    print("==================================================")

    # Load sequence dataset to sample a single sequence for testing
    data_path = os.path.join("data", "processed", "X_sequences.npy")
    
    if os.path.exists(data_path):
        X_all = np.load(data_path)
        sample_seq = X_all[0]  # First sequence (5 time steps, 15 features)
        
        print("\n--- Testing Single Sequence Prediction ---")
        result = predict_sequence(sample_seq)
        
        print(f"Prediction: {result['class_name']}")
        print(f"Infiltration Probability: {result['infiltration_probability']:.2f}")
        print(f"Benign Probability: {result['benign_probability']:.2f}")
        print("------------------------------------------")
    else:
        print(f"No processed dataset found at {data_path} to test standard sample.")