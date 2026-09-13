"""
================================================================================
SIH26153 PROTOTYPE — MITRE ATT&CK INTERPRETATION LAYER
================================================================================
This module interprets predicted network anomalies by analyzing observable 
network-state features over 30-second window sequences.

IMPORTANT SAFETY & TRANSPARENCY NOTE:
- This is an INTERPRETATION LAYER based on deterministic heuristics.
- It provides decision-support context ("behavior consistent with...").
- It does NOT claim that the underlying ML model directly proves a specific 
  MITRE ATT&CK technique with absolute certainty.
- Where feature evidence is inconclusive, it explicitly returns:
  "Insufficient evidence for specific technique"
"""

from typing import Dict, List, Union, Any
import numpy as np

# ==============================================================================
# RULE CONFIGURATION THRESHOLDS
# Thresholds are centrally managed here for easy tuning and transparency.
# ==============================================================================
RULE_CONFIG = {
    # Reconnaissance / Discovery: High destination port fan-out across 30s window
    "RECON_UNIQUE_PORTS_THRESHOLD": 15,
    "RECON_FLOW_COUNT_THRESHOLD": 50,

    # Exfiltration: Sustained high volume outbound data transfer
    "EXFIL_BYTES_PER_SEC_THRESHOLD": 500_000,  # 500 KB/s
    "EXFIL_TOTAL_BYTES_THRESHOLD": 10_000_000,  # 10 MB total sequence volume

    # Lateral Movement: High host-to-host flow counts with low total byte payloads
    "LATERAL_FLOW_COUNT_THRESHOLD": 80,
    "LATERAL_AVG_FORWARD_BYTES_MAX": 150,

    # Command and Control (C2): Beaconing pattern (low packet sizes, fast packet rate)
    "C2_PACKETS_PER_SEC_THRESHOLD": 40.0,
    "C2_AVG_PACKET_LENGTH_MAX": 100.0,
}


# Feature index mapping corresponding to the 15 network-state features
FEATURE_NAMES = [
    "flow_count",                       # 0
    "total_forward_packets",            # 1
    "total_backward_packets",           # 2
    "total_packets",                    # 3
    "total_forward_bytes",              # 4
    "total_backward_bytes",             # 5
    "total_bytes",                      # 6
    "unique_destination_ports",         # 7
    "unique_protocols",                 # 8
    "average_flow_duration",            # 9
    "median_flow_duration",             # 10
    "average_flow_bytes_per_second",    # 11
    "average_flow_packets_per_second",  # 12
    "average_forward_packet_length",    # 13
    "average_backward_packet_length",   # 14
]


def _extract_feature_summary(features: Union[np.ndarray, List[float], Dict[str, float]]) -> Dict[str, float]:
    """
    Normalizes feature input (dict, 1D array, or 2D/3D sequence array) 
    into a simple dictionary of average feature values across the sequence.
    """
    if isinstance(features, dict):
        return features

    feat_arr = np.array(features, dtype=np.float32)

    # Handle sequence input (5, 15) or (1, 5, 15) by taking column averages
    if feat_arr.ndim >= 2:
        feat_arr = np.mean(feat_arr, axis=0)
        if feat_arr.ndim > 1:
            feat_arr = np.mean(feat_arr, axis=0)

    if len(feat_arr) != 15:
        raise ValueError(f"Expected 15 features, got {len(feat_arr)}")

    return {name: float(val) for name, val in zip(FEATURE_NAMES, feat_arr)}


def map_network_state_to_mitre(
    features: Union[np.ndarray, List[float], Dict[str, float]],
    prediction: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Maps observed network-state features and LSTM predictions to MITRE ATT&CK stages.

    Parameters:
        features: Array or dictionary of the 15 network-state features.
        prediction: Dict from predict_sequence containing 'predicted_class', 
                    'class_name', and 'infiltration_probability'.

    Returns:
        Dict containing attack_stage, mitre_tactic, mitre_technique, technique_id, 
        rule_confidence, reason, and supporting_features.
    """
    summary = _extract_feature_summary(features)
    predicted_class = prediction.get("predicted_class", 0)

    # --------------------------------------------------------------------------
    # CASE 1: Normal Traffic (Class 0 predicted)
    # --------------------------------------------------------------------------
    if predicted_class == 0:
        return {
            "attack_stage": "Normal",
            "mitre_tactic": "None",
            "mitre_technique": "None",
            "technique_id": "N/A",
            "rule_confidence": 0.95,
            "reason": "Observed traffic metrics fall within expected normal baseline parameters.",
            "supporting_features": []
        }

    # --------------------------------------------------------------------------
    # CASE 2: Suspicious / Infiltration Traffic (Class 1 predicted)
    # --------------------------------------------------------------------------

    # --- Rule 2A: Reconnaissance / Scanning Pattern ---
    if (summary["unique_destination_ports"] >= RULE_CONFIG["RECON_UNIQUE_PORTS_THRESHOLD"] and
            summary["flow_count"] >= RULE_CONFIG["RECON_FLOW_COUNT_THRESHOLD"]):
        return {
            "attack_stage": "Reconnaissance-like",
            "mitre_tactic": "Discovery",
            "mitre_technique": "Network Service Discovery",
            "technique_id": "T1046",
            "rule_confidence": 0.85,
            "reason": (
                f"Behavior is consistent with active network scanning. High unique destination ports "
                f"({summary['unique_destination_ports']:.0f}) and elevated flow count "
                f"({summary['flow_count']:.0f}) observed."
            ),
            "supporting_features": ["unique_destination_ports", "flow_count"]
        }

    # --- Rule 2B: Exfiltration (High Outbound Volume) ---
    if (summary["average_flow_bytes_per_second"] >= RULE_CONFIG["EXFIL_BYTES_PER_SEC_THRESHOLD"] or
            summary["total_bytes"] >= RULE_CONFIG["EXFIL_TOTAL_BYTES_THRESHOLD"]):
        return {
            "attack_stage": "Exfiltration",
            "mitre_tactic": "Exfiltration",
            "mitre_technique": "Exfiltration Over Unencrypted Network Medium",
            "technique_id": "T1048",
            "rule_confidence": 0.80,
            "reason": (
                f"Traffic suggests high-volume outbound data transfer. Average rate "
                f"({summary['average_flow_bytes_per_second'] / 1024:.1f} KB/s) and total volume "
                f"({summary['total_bytes'] / (1024 * 1024):.1f} MB) exceed baseline thresholds."
            ),
            "supporting_features": ["average_flow_bytes_per_second", "total_bytes", "total_forward_bytes"]
        }

    # --- Rule 2C: Command and Control (Beaconing Activity) ---
    if (summary["average_flow_packets_per_second"] >= RULE_CONFIG["C2_PACKETS_PER_SEC_THRESHOLD"] and
            summary["average_forward_packet_length"] <= RULE_CONFIG["C2_AVG_PACKET_LENGTH_MAX"]):
        return {
            "attack_stage": "Command and Control",
            "mitre_tactic": "Command and Control",
            "mitre_technique": "Application Layer Protocol: Beaconing",
            "technique_id": "T1071",
            "rule_confidence": 0.75,
            "reason": (
                f"High-frequency small-packet traffic pattern resembles Automated C2 Beaconing. "
                f"Packet rate ({summary['average_flow_packets_per_second']:.1f} pkt/s) with small avg payload "
                f"({summary['average_forward_packet_length']:.1f} bytes)."
            ),
            "supporting_features": ["average_flow_packets_per_second", "average_forward_packet_length"]
        }

    # --- Rule 2D: Lateral Movement / Internal Discovery ---
    if (summary["flow_count"] >= RULE_CONFIG["LATERAL_FLOW_COUNT_THRESHOLD"] and
            summary["average_forward_packet_length"] <= RULE_CONFIG["LATERAL_AVG_FORWARD_BYTES_MAX"]):
        return {
            "attack_stage": "Lateral Movement",
            "mitre_tactic": "Lateral Movement",
            "mitre_technique": "Remote Services / Internal Host Enumeration",
            "technique_id": "T1021",
            "rule_confidence": 0.70,
            "reason": (
                f"Unusually high flow frequency ({summary['flow_count']:.0f}) with low payload size "
                f"indicates potential internal host enumeration or lateral connection attempts."
            ),
            "supporting_features": ["flow_count", "average_forward_packet_length"]
        }

    # --- Fallback: Generic Anomaly ---
    return {
        "attack_stage": "Initial Access-like",
        "mitre_tactic": "Initial Access",
        "mitre_technique": "Insufficient evidence for specific technique",
        "technique_id": "None",
        "rule_confidence": 0.50,
        "reason": (
            "Observed network features indicate anomalous activity but do not provide enough "
            "evidence to identify a specific ATT&CK technique."
        ),
        "supporting_features": ["flow_count", "average_flow_duration"]
    }


# ==============================================================================
# DEMO EXECUTION & SYNTHETIC TEST SUITE
# ==============================================================================
if __name__ == "__main__":
    print("==========================================================================")
    print("      SIH26153 PROTOTYPE — MITRE ATT&CK MAPPING TEST SUITE               ")
    print("==========================================================================")

    # Synthetic Test Scenarios (15 features each)
    test_cases = [
        {
            "name": "Normal Baseline Traffic",
            "pred": {"predicted_class": 0, "class_name": "Benign", "infiltration_probability": 0.05},
            # Low ports, low flow count, normal sizes
            "features": [10, 20, 20, 40, 500, 500, 1000, 2, 1, 1.5, 1.2, 500, 10, 50, 50]
        },
        {
            "name": "Reconnaissance / Scanning Pattern",
            "pred": {"predicted_class": 1, "class_name": "Infiltration", "infiltration_probability": 0.88},
            # 35 unique ports, 120 flows -> triggers Network Service Discovery
            "features": [120, 150, 10, 160, 2000, 100, 2100, 35, 2, 0.5, 0.2, 1000, 50, 30, 10]
        },
        {
            "name": "Exfiltration Pattern",
            "pred": {"predicted_class": 1, "class_name": "Infiltration", "infiltration_probability": 0.94},
            # High byte volume rate -> triggers Exfiltration
            "features": [15, 5000, 200, 5200, 15_000_000, 5000, 15_005_000, 1, 1, 10.0, 9.5, 800_000, 50, 1500, 50]
        },
        {
            "name": "Command and Control (C2) Beaconing Pattern",
            "pred": {"predicted_class": 1, "class_name": "Infiltration", "infiltration_probability": 0.79},
            # High packet rate with very low payload -> triggers C2 Beaconing
            "features": [30, 2000, 2000, 4000, 80000, 80000, 160000, 1, 1, 0.1, 0.1, 5000, 60, 40, 40]
        },
        {
            "name": "Generic Anomaly (Unclear Specific Signature)",
            "pred": {"predicted_class": 1, "class_name": "Infiltration", "infiltration_probability": 0.65},
            # Moderate numbers that trigger prediction but no specific technique rule
            "features": [25, 30, 30, 60, 2000, 2000, 4000, 5, 1, 3.0, 2.5, 1000, 15, 80, 80]
        }
    ]

    for test in test_cases:
        print(f"\n--- Scenario: {test['name']} ---")
        result = map_network_state_to_mitre(test["features"], test["pred"])
        print(f"Prediction Input : {test['pred']['class_name']} (Prob: {test['pred']['infiltration_probability']})")
        print(f"Attack Stage     : {result['attack_stage']}")
        print(f"MITRE Tactic     : {result['mitre_tactic']}")
        print(f"MITRE Technique  : {result['mitre_technique']} ({result['technique_id']})")
        print(f"Rule Confidence  : {result['rule_confidence']:.2f}")
        print(f"Reason           : {result['reason']}")
        print(f"Supporting Feats : {result['supporting_features']}")

    print("\n==========================================================================")
    print("Mapping test suite execution complete.")