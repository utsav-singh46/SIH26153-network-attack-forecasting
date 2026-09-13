import os
import joblib
import numpy as np
import torch
import torch.nn as nn

# Import the standalone MITRE ATT&CK interpretation module
from mitre_mapping import map_network_state_to_mitre

# =====================================================================
# DEFAULT PATH DEFINITIONS
# =====================================================================
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
    Loads PyTorch LSTM model weights from disk and sets to evaluation mode.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    model = InferenceLSTMModel(input_size=15, hidden_size=64, num_layers=1, num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def load_scaler(scaler_path: str = DEFAULT_SCALER_PATH):
    """
    Loads the fitted StandardScaler from disk.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler pickle file not found at: {scaler_path}")
    
    return joblib.load(scaler_path)


# =====================================================================
# DETERMINISTIC OVERALL ASSESSMENT LOGIC
# =====================================================================
def determine_overall_assessment(predicted_class: int, attack_stage: str) -> str:
    """
    Combines independent ML prediction and Rule-based behavior stage into an overall assessment.

    Rules:
    1. ML says Benign (0) AND MITRE stage is "Normal" -> "Normal"
    2. ML says Benign (0) BUT MITRE stage is non-Normal -> "Suspicious Behaviour"
    3. ML says Infiltration (1) AND MITRE stage is non-Normal -> "Likely Infiltration"
    4. ML says Infiltration (1) AND MITRE stage is "Normal" -> "Likely Infiltration"
    """
    if predicted_class == 0 and attack_stage == "Normal":
        return "Normal"
    elif predicted_class == 0 and attack_stage != "Normal":
        return "Suspicious Behaviour"
    elif predicted_class == 1:
        return "Likely Infiltration"
    
    return "Normal"


# =====================================================================
# CORE INTEGRATED PREDICTION FUNCTIONS
# =====================================================================
def predict_sequence(
    sequence: np.ndarray,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH
) -> dict:
    """
    Runs independent ML prediction and rule-based MITRE ATT&CK mapping on a SINGLE sequence input.
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

    # 2. Preprocess Input Sequence for LSTM Inference
    scaler = load_scaler(scaler_path)
    seq_flat = seq.reshape(-1, 15)
    seq_scaled_flat = scaler.transform(seq_flat)
    seq_scaled = seq_scaled_flat.reshape(1, 5, 15)

    # 3. Perform ML Model Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(checkpoint_path, device=device.type)
    tensor_input = torch.tensor(seq_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(torch.argmax(logits, dim=1).cpu().numpy()[0])

    benign_prob = float(probabilities[0])
    infiltration_prob = float(probabilities[1])

    ml_prediction = {
        "predicted_class": predicted_class,
        "class_name": CLASS_MAP[predicted_class],
        "benign_probability": round(benign_prob, 4),
        "infiltration_probability": round(infiltration_prob, 4),
    }

    # 4. ALWAYS Run Behavior Mapping independently on raw features
    # Force mock 'Infiltration' prediction into rule mapper so heuristics evaluate feature metrics independently
    mock_prediction_for_rules = {"predicted_class": 1}
    mitre_interpretation = map_network_state_to_mitre(seq[0], mock_prediction_for_rules)

    # 5. Calculate Overall Assessment
    overall_assessment = determine_overall_assessment(
        predicted_class=ml_prediction["predicted_class"],
        attack_stage=mitre_interpretation["attack_stage"]
    )

    # 6. Combine ML Metrics, Rule Interpretations, and Assessment into Output
    combined_result = {
        "predicted_class": ml_prediction["predicted_class"],
        "class_name": ml_prediction["class_name"],
        "benign_probability": ml_prediction["benign_probability"],
        "infiltration_probability": ml_prediction["infiltration_probability"],
        "overall_assessment": overall_assessment,
        "attack_stage": mitre_interpretation["attack_stage"],
        "mitre_tactic": mitre_interpretation["mitre_tactic"],
        "mitre_technique": mitre_interpretation["mitre_technique"],
        "technique_id": mitre_interpretation["technique_id"],
        "rule_confidence": mitre_interpretation["rule_confidence"],
        "reason": mitre_interpretation["reason"],
        "supporting_features": mitre_interpretation["supporting_features"],
    }

    return combined_result


def predict_batch(
    sequences: np.ndarray,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH
) -> list:
    """
    Runs independent ML prediction and rule-based MITRE ATT&CK mapping on MULTIPLE sequence inputs.
    """
    seqs = np.array(sequences, dtype=np.float32)
    if seqs.ndim != 3 or seqs.shape[1:] != (5, 15):
        raise ValueError(f"Expected batch shape (N, 5, 15), got {seqs.shape}")

    n_samples = seqs.shape[0]

    # Preprocess batch
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

        ml_pred = {
            "predicted_class": pred_cls,
            "class_name": CLASS_MAP[pred_cls],
            "benign_probability": round(ben_prob, 4),
            "infiltration_probability": round(inf_prob, 4),
        }

        # ALWAYS run behavior mapper independently
        mock_pred = {"predicted_class": 1}
        mitre_interp = map_network_state_to_mitre(seqs[i], mock_pred)

        overall_assessment = determine_overall_assessment(
            predicted_class=pred_cls,
            attack_stage=mitre_interp["attack_stage"]
        )

        results.append({
            "sample_index": i,
            "predicted_class": ml_pred["predicted_class"],
            "class_name": ml_pred["class_name"],
            "benign_probability": ml_pred["benign_probability"],
            "infiltration_probability": ml_pred["infiltration_probability"],
            "overall_assessment": overall_assessment,
            "attack_stage": mitre_interp["attack_stage"],
            "mitre_tactic": mitre_interp["mitre_tactic"],
            "mitre_technique": mitre_interp["mitre_technique"],
            "technique_id": mitre_interp["technique_id"],
            "rule_confidence": mitre_interp["rule_confidence"],
            "reason": mitre_interp["reason"],
            "supporting_features": mitre_interp["supporting_features"],
        })

    return results


# =====================================================================
# INTEGRATED PIPELINE DEMO & 5-SCENARIO TEST SUITE
# =====================================================================
if __name__ == "__main__":
    print("==========================================================================")
    print("      SIH26153 PROTOTYPE — INTEGRATED PREDICTION & RULE MAPPING            ")
    print("==========================================================================")

    data_path = os.path.join("data", "processed", "X_sequences.npy")

    # Construct the 5 required test cases
    test_cases = []

    # Test Case 1: Real dataset sequence
    if os.path.exists(data_path):
        X_all = np.load(data_path)
        test_cases.append({
            "name": "Test 1: Normal Real Dataset Sequence (Sample 0)",
            "sequence": X_all[0]
        })
    else:
        # Fallback normal synthetic sequence
        test_cases.append({
            "name": "Test 1: Normal Real Dataset Sequence (Fallback Synthetic)",
            "sequence": np.tile([10, 20, 20, 40, 500, 500, 1000, 2, 1, 1.5, 1.2, 500, 10, 50, 50], (5, 1))
        })

    # Test Case 2: Synthetic Scanning Sequence
    seq_scan = np.zeros((5, 15), dtype=np.float32)
    seq_scan[:, 0] = 120.0  # flow_count
    seq_scan[:, 7] = 35.0   # unique_destination_ports
    seq_scan[:, 11] = 1000.0
    test_cases.append({
        "name": "Test 2: Synthetic Scanning Sequence",
        "sequence": seq_scan
    })

    # Test Case 3: Synthetic Exfiltration Sequence
    seq_exfil = np.zeros((5, 15), dtype=np.float32)
    seq_exfil[:, 0] = 15.0
    seq_exfil[:, 6] = 15_005_000.0  # total_bytes
    seq_exfil[:, 11] = 800_000.0    # average_flow_bytes_per_second
    test_cases.append({
        "name": "Test 3: Synthetic Exfiltration Sequence",
        "sequence": seq_exfil
    })

    # Test Case 4: Synthetic C2 Sequence
    seq_c2 = np.zeros((5, 15), dtype=np.float32)
    seq_c2[:, 0] = 30.0
    seq_c2[:, 12] = 60.0   # average_flow_packets_per_second
    seq_c2[:, 13] = 40.0   # average_forward_packet_length
    test_cases.append({
        "name": "Test 4: Synthetic C2 Sequence",
        "sequence": seq_c2
    })

    # Test Case 5: Generic Suspicious Anomaly
    seq_anomaly = np.zeros((5, 15), dtype=np.float32)
    seq_anomaly[:, 0] = 25.0
    seq_anomaly[:, 7] = 5.0
    seq_anomaly[:, 9] = 3.0
    seq_anomaly[:, 11] = 1000.0
    test_cases.append({
        "name": "Test 5: Generic Suspicious Anomaly",
        "sequence": seq_anomaly
    })

    # Execute all 5 scenarios
    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        res = predict_sequence(test["sequence"])

        print(f"ML Prediction          : {res['class_name']} (Infiltration Prob: {res['infiltration_probability']:.2f}, Benign Prob: {res['benign_probability']:.2f})")
        print(f"Overall Assessment     : {res['overall_assessment']}")
        print(f"Attack Stage           : {res['attack_stage']}")
        print(f"MITRE Tactic           : {res['mitre_tactic']}")
        print(f"MITRE Technique        : {res['mitre_technique']}")
        print(f"Technique ID           : {res['technique_id']}")
        print(f"Rule Confidence        : {res['rule_confidence']:.2f}")
        print(f"Reason                 : {res['reason']}")
        print(f"Supporting Features    : {res['supporting_features']}")

    print("\n==========================================================================")
    print("Integrated prediction pipeline execution complete.")