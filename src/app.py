"""
================================================================================
SIH26153 — AI NETWORK ATTACK FORECASTING CONSOLE
================================================================================
State-machine driven monitoring dashboard with scenario-tailored demo metrics,
unified technical details, and dynamic chronological replay graphing.

Design & Functional Rules:
- Starts in a strictly NORMAL baseline state (0% Infiltration Chance / 100% No Attack).
- Single-button "START ATTACK DEMO" trigger opens a selection modal dialog ("CHOOSE DEMO").
- Unified Source of Truth: In Demo Mode, all sections (Status, Popup, Technical Details,
  Why We Flagged It) draw strictly from SCENARIO_DEFINITIONS.
- Analysis Mode fallback: Uses predict_sequence() when no active demo is running.
- Dynamic Graph Progression: Uses st.session_state.demo_history and st.line_chart with
  chronological x-axis (T-120s -> T-90s -> T-60s -> T-30s -> NOW on far right).
- Uses ONLY standard standard-library / base packages (numpy, pandas, streamlit).
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st

# Direct import of backend inference module (UNTOUCHED)
from predict import predict_sequence

# ==============================================================================
# BASELINE NORMAL SEQUENCE GENERATOR
# ==============================================================================
def get_baseline_normal_sequence() -> np.ndarray:
    """Returns a purely normal baseline sequence (5 windows x 15 features)."""
    base_window = np.array([10, 20, 20, 40, 500, 500, 1000, 2, 1, 1.5, 1.2, 500, 10, 50, 50], dtype=np.float32)
    return np.tile(base_window, (5, 1))


def _build_steps_for_scenario(anom_w3: np.ndarray, anom_w4: np.ndarray) -> list:
    """Helper to build 5 chronological sequence windows progressing from normal to anomaly."""
    base = get_baseline_normal_sequence()
    steps = []

    # Step 0: Pure Normal Baseline
    steps.append(base.copy())

    # Step 1: Pure Normal Baseline
    steps.append(base.copy())

    # Step 2: Slight Activity Increase
    s2 = base.copy()
    s2[4, 0] = 18.0
    s2[4, 6] = 2000.0
    steps.append(s2)

    # Step 3: Elevated Traffic
    s3 = base.copy()
    s3[3, 0] = anom_w3[0]
    s3[3, 7] = anom_w3[7]
    s3[3, 6] = anom_w3[6]
    s3[3, 11] = anom_w3[11]
    s3[3, 12] = anom_w3[12]
    s3[3, 13] = anom_w3[13]
    s3[3, 9] = anom_w3[9]
    steps.append(s3)

    # Step 4: Target Anomaly Manifestation
    s4 = base.copy()
    s4[3] = anom_w3
    s4[4] = anom_w4
    steps.append(s4)

    return steps


# ==============================================================================
# EXPLICIT SCENARIO DEFINITIONS & UNIFIED TELEMETRY
# ==============================================================================
_base_w = np.array([10, 20, 20, 40, 500, 500, 1000, 2, 1, 1.5, 1.2, 500, 10, 50, 50], dtype=np.float32)

_scan_w3 = _base_w.copy(); _scan_w3[0], _scan_w3[7] = 60.0, 18.0
_scan_w4 = _base_w.copy(); _scan_w4[0], _scan_w4[7] = 120.0, 35.0

_exfil_w3 = _base_w.copy(); _exfil_w3[6], _exfil_w3[11] = 5_000_000.0, 300_000.0
_exfil_w4 = _base_w.copy(); _exfil_w4[6], _exfil_w4[11] = 15_005_000.0, 800_000.0

_c2_w3 = _base_w.copy(); _c2_w3[12], _c2_w3[13] = 30.0, 45.0
_c2_w4 = _base_w.copy(); _c2_w4[12], _c2_w4[13] = 65.0, 35.0

_susp_w3 = _base_w.copy(); _susp_w3[0], _susp_w3[9] = 20.0, 2.5
_susp_w4 = _base_w.copy(); _susp_w4[0], _susp_w4[9] = 28.0, 3.2


SCENARIO_DEFINITIONS = {
    "Network Scanning": {
        "title": "SCANNING ACTIVITY DETECTED",
        "what_happened": "The system noticed one device contacting many different network ports in a short period of time.",
        "evidence": [
            "35 different destination ports were contacted.",
            "120 network connections were observed."
        ],
        "attack_type": "Network Scanning",
        "mitre_tactic": "Discovery",
        "mitre_technique": "Network Service Discovery",
        "technique_id": "T1046",
        "evidence_strength": 0.85,
        "supporting_features": ["unique_destination_ports", "flow_count"],
        "demo_status": "SUSPICIOUS",
        "demo_prob_attack": 0.78,
        "demo_prob_no_attack": 0.22,
        "demo_stage": "Reconnaissance-like",
        "immediate_actions": [
            "Identify the source host generating the scan.",
            "Review firewall and network-security logs for affected destinations.",
            "Rate-limit or block unauthorized scanning traffic according to security policy."
        ],
        "prevention_actions": [
            "Restrict unnecessary exposed services and ports.",
            "Review network segmentation so internal systems are not broadly reachable.",
            "Investigate whether the source host shows other signs of compromise."
        ],
        "sequence_steps": _build_steps_for_scenario(_scan_w3, _scan_w4)
    },
    "Data Exfiltration": {
        "title": "POSSIBLE DATA TRANSFER DETECTED",
        "what_happened": "The system noticed an unusually large amount of data leaving the network.",
        "evidence": [
            "High outbound data volume was observed.",
            "Data transfer remained high for the current period."
        ],
        "attack_type": "Possible Data Exfiltration",
        "mitre_tactic": "Exfiltration",
        "mitre_technique": "Exfiltration Over Unencrypted Network Medium",
        "technique_id": "T1048",
        "evidence_strength": 0.80,
        "supporting_features": [
            "average_flow_bytes_per_second",
            "total_bytes",
            "total_forward_bytes"
        ],
        "demo_status": "HIGH RISK",
        "demo_prob_attack": 0.92,
        "demo_prob_no_attack": 0.08,
        "demo_stage": "Exfiltration",
        "immediate_actions": [
            "Identify and verify the destination receiving the outbound traffic.",
            "Verify whether the transfer is authorized.",
            "Restrict or quarantine the affected endpoint if malicious transfer is confirmed."
        ],
        "prevention_actions": [
            "Review outbound firewall, proxy, and DNS logs.",
            "Apply egress controls to limit unnecessary external destinations.",
            "Identify sensitive data stores and monitor unusual outbound transfers."
        ],
        "sequence_steps": _build_steps_for_scenario(_exfil_w3, _exfil_w4)
    },
    "Command & Control": {
        "title": "POSSIBLE ATTACKER COMMUNICATION",
        "what_happened": "The system noticed repeated small outbound connections that resemble automated communication.",
        "evidence": [
            "Repeated outbound connections.",
            "High packet frequency with small packets."
        ],
        "attack_type": "Possible Command & Control",
        "mitre_tactic": "Command and Control",
        "mitre_technique": "Application Layer Protocol: Beaconing",
        "technique_id": "T1071",
        "evidence_strength": 0.75,
        "supporting_features": [
            "average_flow_packets_per_second",
            "average_forward_packet_length"
        ],
        "demo_status": "HIGH RISK",
        "demo_prob_attack": 0.88,
        "demo_prob_no_attack": 0.12,
        "demo_stage": "Command and Control",
        "immediate_actions": [
            "Identify the destination associated with repeated outbound connections.",
            "Investigate the affected endpoint for unauthorized processes or malware.",
            "Isolate the endpoint from the network if malicious activity is confirmed."
        ],
        "prevention_actions": [
            "Review DNS, proxy, and firewall logs for repeated communication patterns.",
            "Restrict unnecessary outbound connections.",
            "Verify that endpoint protections and security definitions are up to date."
        ],
        "sequence_steps": _build_steps_for_scenario(_c2_w3, _c2_w4)
    },
    "Suspicious Activity": {
        "title": "SUSPICIOUS NETWORK ACTIVITY",
        "what_happened": "The network behaviour looks unusual, but there is not enough evidence to identify one specific attack type.",
        "evidence": [
            "A large number of network connections were observed.",
            "Flow duration was different from the normal baseline."
        ],
        "attack_type": "Suspicious Network Activity",
        "mitre_tactic": "Initial Access",
        "mitre_technique": "Insufficient evidence for a specific technique",
        "technique_id": "None",
        "evidence_strength": 0.50,
        "supporting_features": [
            "flow_count",
            "average_flow_duration"
        ],
        "demo_status": "SUSPICIOUS",
        "demo_prob_attack": 0.61,
        "demo_prob_no_attack": 0.39,
        "demo_stage": "Initial Access-like",
        "immediate_actions": [
            "Identify the affected host and review surrounding traffic.",
            "Compare current activity against the host's normal baseline.",
            "Review endpoint, firewall, and authentication logs for related events."
        ],
        "prevention_actions": [
            "Investigate whether the anomaly is caused by legitimate workload changes.",
            "Correlate activity with other security alerts.",
            "Maintain monitoring until traffic returns to baseline."
        ],
        "sequence_steps": _build_steps_for_scenario(_susp_w3, _susp_w4)
    }
}


# ==============================================================================
# PAGE CONFIGURATION & SANS-SERIF STYLING
# ==============================================================================
st.set_page_config(
    page_title="SIH26153 — AI Network Attack Forecasting",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Global Page Styling: Sans-Serif Typography & Pure Dark Background */
    html, body, [class*="css"], .stApp {
        background-color: #0A0A0A !important;
        color: #F0F0F0 !important;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif !important;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Clean Header */
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #888888;
        margin-bottom: 24px;
        font-weight: 400;
    }
    
    /* Status Section Styling */
    .status-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888888;
        margin-bottom: 8px;
    }
    
    .status-val-normal {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        margin-bottom: 12px;
    }
    .status-val-alert {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF3B30;
        letter-spacing: -0.5px;
        margin-bottom: 12px;
    }
    
    .status-explanation {
        font-size: 0.95rem;
        color: #CCCCCC;
        margin-top: 8px;
        margin-bottom: 24px;
    }
    
    /* Section Division Line */
    .divider {
        border: none;
        border-top: 1px solid #1F1F1F;
        margin: 28px 0;
    }

    /* Attack Forecast Progression Steps */
    .stage-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    .stage-pill {
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        background-color: #141414;
        border: 1px solid #222222;
        color: #666666;
    }
    .stage-pill-active-normal {
        background-color: #1A1A1A;
        border: 1px solid #444444;
        color: #FFFFFF;
        font-weight: 600;
    }
    .stage-pill-active-alert {
        background-color: #2C0B0B;
        border: 1px solid #FF3B30;
        color: #FF3B30;
        font-weight: 600;
    }

    .reason-bullet {
        font-size: 0.95rem;
        color: #DDDDDD;
        margin-bottom: 8px;
        line-height: 1.5;
    }
    
    .tech-table {
        width: 100%;
        font-size: 0.85rem;
        color: #CCCCCC;
        border-collapse: collapse;
    }
    .tech-table td {
        padding: 6px 0;
        border-bottom: 1px solid #1A1A1A;
    }
    .tech-table td:first-child {
        color: #888888;
        width: 40%;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #141414 !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 10px 16px !important;
    }
    .stButton>button:hover {
        background-color: #222222 !important;
        border-color: #555555 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# MODAL DIALOGS
# ==============================================================================
@st.dialog("CHOOSE DEMO")
def render_demo_selection_dialog():
    """Renders modal selection for picking an attack demo scenario."""
    scenario_choice = st.radio(
        "Select Attack Scenario:",
        ["Network Scanning", "Data Exfiltration", "Command & Control", "Suspicious Activity"],
        index=0
    )

    if st.button("START"):
        st.session_state.demo_scenario = scenario_choice
        st.session_state.demo_steps_data = SCENARIO_DEFINITIONS[scenario_choice]["sequence_steps"]
        st.session_state.demo_step = 0
        st.session_state.demo_running = True
        st.session_state.demo_alert_active = False
        st.session_state.demo_complete = False
        st.session_state.demo_history = []  # Reset graph history cleanly
        st.rerun()


@st.dialog("⚠️ ATTACK ALERT")
def render_attack_alert_dialog(scenario_key: str):
    """
    Renders attack alert details derived strictly from the explicit scenario definition.
    Guarantees professional SOC guidance and no technical contradictions.
    """
    spec = SCENARIO_DEFINITIONS.get(scenario_key, SCENARIO_DEFINITIONS["Suspicious Activity"])

    st.markdown(f"### {spec['title']}")
    st.write(spec["what_happened"])

    st.markdown("### EVIDENCE")
    for ev in spec["evidence"]:
        st.write(f"• {ev}")

    st.markdown("### ATTACK TYPE")
    st.write(f"**{spec['attack_type']}**")

    st.markdown("### MITRE ATT&CK")
    st.write(f"**Tactic:** {spec['mitre_tactic']}")
    st.write(f"**Technique:** {spec['mitre_technique']}")
    if spec["technique_id"] and spec["technique_id"] != "None":
        st.write(f"**ID:** `{spec['technique_id']}`")
    else:
        st.write("**ID:** None")

    st.markdown("### RECOMMENDED RESPONSE")
    st.caption("Immediate actions:")
    for act in spec["immediate_actions"]:
        st.write(f"• {act}")

    st.markdown("### PREVENTION")
    st.caption("Follow-up / prevention:")
    for prev in spec["prevention_actions"]:
        st.write(f"• {prev}")

    if st.button("CLOSE"):
        st.session_state.demo_alert_active = False
        st.rerun()


# ==============================================================================
# SESSION STATE MANAGEMENT
# ==============================================================================
def initialize_session_state():
    """Ensures clean initial dashboard state on fresh page load."""
    if "demo_running" not in st.session_state:
        st.session_state.demo_running = False
    if "demo_step" not in st.session_state:
        st.session_state.demo_step = 0
    if "demo_scenario" not in st.session_state:
        st.session_state.demo_scenario = None
    if "demo_steps_data" not in st.session_state:
        st.session_state.demo_steps_data = None
    if "demo_alert_active" not in st.session_state:
        st.session_state.demo_alert_active = False
    if "demo_complete" not in st.session_state:
        st.session_state.demo_complete = False
    if "demo_history" not in st.session_state:
        st.session_state.demo_history = []


# ==============================================================================
# MAIN DASHBOARD LAYOUT & EXECUTION PIPELINE
# ==============================================================================
def main():
    initialize_session_state()

    # --------------------------------------------------------------------------
    # HEADER & DEMO TRIGGER
    # --------------------------------------------------------------------------
    st.markdown('<div class="app-title">SIH26153</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">AI Network Attack Forecasting</div>', unsafe_allow_html=True)

    col_btn, col_note = st.columns([2, 1.2])
    with col_btn:
        if st.button("START ATTACK DEMO"):
            render_demo_selection_dialog()
    with col_note:
        if st.session_state.demo_scenario and (st.session_state.demo_running or st.session_state.demo_complete):
            st.markdown("<div style='padding-top:10px; font-size:0.8rem; color:#888888;'>Demo replay active</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # REPLAY STATE MACHINE PROGRESSION & HISTORY COLLECTION
    # --------------------------------------------------------------------------
    if st.session_state.demo_running and not st.session_state.demo_alert_active:
        step_idx = st.session_state.demo_step

        if step_idx >= len(st.session_state.demo_steps_data) - 1:
            st.session_state.demo_running = False
            st.session_state.demo_complete = True
            st.session_state.demo_alert_active = True
        else:
            time.sleep(0.8)  # ~1s per 30s sim window delay
            st.session_state.demo_step += 1
            st.rerun()

    # Determine active sequence window
    if st.session_state.demo_steps_data and (st.session_state.demo_running or st.session_state.demo_complete or st.session_state.demo_alert_active):
        step_idx = min(st.session_state.demo_step, len(st.session_state.demo_steps_data) - 1)
        current_sequence = st.session_state.demo_steps_data[step_idx]
        is_demo_mode = True
        
        # Build chronological history up to current step for active scenario
        st.session_state.demo_history = [
            st.session_state.demo_steps_data[i][-1, 0] for i in range(step_idx + 1)
        ]
    else:
        # Default Initial Analysis Mode -> Pure Normal Baseline
        current_sequence = get_baseline_normal_sequence()
        is_demo_mode = False

    # Backend inference for technical details (UNTOUCHED)
    res = predict_sequence(current_sequence)

    # Determine display metrics & status
    if not is_demo_mode or (st.session_state.demo_step == 0 and st.session_state.demo_running):
        prob_inf = 0.0
        prob_benign = 1.0
        status_text = "NORMAL"
        raw_stage = "Normal"
        is_alert = False
    else:
        scen_key = st.session_state.demo_scenario
        scen_spec = SCENARIO_DEFINITIONS.get(scen_key, SCENARIO_DEFINITIONS["Suspicious Activity"])
        
        if st.session_state.demo_step >= 3:
            status_text = scen_spec["demo_status"]
            prob_inf = scen_spec["demo_prob_attack"]
            prob_benign = scen_spec["demo_prob_no_attack"]
            raw_stage = scen_spec["demo_stage"]
            is_alert = True
        else:
            status_text = "NORMAL"
            prob_inf = 0.0
            prob_benign = 1.0
            raw_stage = "Normal"
            is_alert = False

    # --------------------------------------------------------------------------
    # NETWORK STATUS
    # --------------------------------------------------------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="status-label">NETWORK STATUS</div>', unsafe_allow_html=True)

    status_class = "status-val-alert" if is_alert else "status-val-normal"
    st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Chance of an Attack", value=f"{prob_inf * 100:.0f}%")
    with c2:
        st.metric(label="Chance of No Attack", value=f"{prob_benign * 100:.0f}%")

    explanation = "Some network activity looks unusual and requires attention." if is_alert else "Network activity is operating within normal parameters."
    st.markdown(f'<div class="status-explanation">{explanation}</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # RENDER EXPLICIT ALERT MODAL (PAUSES PROGRESSION WHILE OPEN)
    # --------------------------------------------------------------------------
    if is_alert and st.session_state.demo_alert_active:
        render_attack_alert_dialog(st.session_state.demo_scenario)

    # --------------------------------------------------------------------------
    # WHAT MAY BE HAPPENING
    # --------------------------------------------------------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="status-label">WHAT MAY BE HAPPENING</div>', unsafe_allow_html=True)

    stage_flow = [
        ("Normal", "Normal"),
        ("Reconnaissance-like", "Scanning"),
        ("Initial Access-like", "Possible Break-in Attempt"),
        ("Exfiltration", "Possible Data Transfer")
    ]

    pills_html = '<div class="stage-container">'
    for internal_key, display_label in stage_flow:
        if raw_stage == internal_key:
            active_cls = "stage-pill-active-normal" if internal_key == "Normal" else "stage-pill-active-alert"
            pills_html += f'<div class="stage-pill {active_cls}">● {display_label}</div>'
        else:
            pills_html += f'<div class="stage-pill">{display_label}</div>'
    pills_html += '</div>'

    st.markdown(pills_html, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # WHY WE FLAGGED IT (DEMO MODE vs ANALYSIS MODE SOURCE OF TRUTH)
    # --------------------------------------------------------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="status-label">WHY WE FLAGGED IT</div>', unsafe_allow_html=True)

    if is_demo_mode and is_alert:
        scen_spec = SCENARIO_DEFINITIONS.get(st.session_state.demo_scenario, SCENARIO_DEFINITIONS["Suspicious Activity"])
        for ev in scen_spec["evidence"]:
            st.markdown(f'<div class="reason-bullet">• {ev}</div>', unsafe_allow_html=True)
    elif is_alert:
        supporting = res.get("supporting_features", [])
        for feat in supporting:
            st.markdown(f'<div class="reason-bullet">• {feat.replace("_", " ").title()} exceeded baseline threshold.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="reason-bullet" style="color: #888888;">• No suspicious traffic patterns detected.</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # RECENT NETWORK ACTIVITY (CHRONOLOGICAL GRAPH: OLD -> NEW / RIGHTMOST NOW)
    # --------------------------------------------------------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="status-label">RECENT NETWORK ACTIVITY</div>', unsafe_allow_html=True)

    labels = ["T-120s", "T-90s", "T-60s", "T-30s", "NOW"]

    if is_demo_mode and len(st.session_state.demo_history) > 0:
        hist = st.session_state.demo_history[-5:]
        active_labels = labels[:len(hist)]
        chart_df = pd.DataFrame({"Traffic Flow Volume": hist}, index=active_labels)
    else:
        # Fallback baseline static window
        flow_counts = current_sequence[:, 0]
        chart_df = pd.DataFrame({"Traffic Flow Volume": flow_counts}, index=labels)

    st.line_chart(chart_df, height=180, color="#FF3B30" if is_alert else "#FFFFFF")

    # --------------------------------------------------------------------------
    # TECHNICAL DETAILS (BUG 1 FIX: UNIFIED SINGLE SOURCE OF TRUTH)
    # --------------------------------------------------------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.expander("Technical Details"):
        if is_demo_mode and st.session_state.demo_scenario:
            scen_spec = SCENARIO_DEFINITIONS.get(st.session_state.demo_scenario, SCENARIO_DEFINITIONS["Suspicious Activity"])
            tactic = scen_spec["mitre_tactic"]
            tech_name = scen_spec["mitre_technique"]
            tech_id = scen_spec["technique_id"]
            evidence_str = f"{scen_spec['evidence_strength']:.2f}"
            supporting_feats = ", ".join(scen_spec["supporting_features"])
        else:
            tactic = res.get('mitre_tactic', 'None')
            tech_name = res.get('mitre_technique', 'None')
            tech_id = res.get('technique_id', 'None')
            evidence_str = f"{res.get('rule_confidence', 0.0):.2f}"
            supporting = res.get("supporting_features", [])
            supporting_feats = ", ".join(supporting) if supporting else "None"

        if tech_name == "Insufficient evidence for specific technique":
            tech_display = "We cannot identify the exact attack type yet"
        else:
            tech_display = tech_name

        st.markdown(
            f"""
            <table class="tech-table">
                <tr><td>MITRE Tactic</td><td><b>{tactic}</b></td></tr>
                <tr><td>MITRE Technique</td><td><b>{tech_display}</b></td></tr>
                <tr><td>Technique ID</td><td><code style="background:#1A1A1A; padding:2px 6px; border-radius:3px;">{tech_id}</code></td></tr>
                <tr><td>Evidence Strength</td><td><b>{evidence_str}</b></td></tr>
                <tr><td>Supporting Features</td><td><code>{supporting_feats}</code></td></tr>
            </table>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()