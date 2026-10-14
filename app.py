import os
import tempfile
from typing import Dict, Tuple
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import pipeline

# ==============================================================================
# Page Configuration & Institutional Banking Styling (Fixed Dimensions, No Emojis)
# ==============================================================================
st.set_page_config(
    page_title="Standard Chartered | Voice Triage System",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Hide Streamlit default header/footer clutter for a clean banking UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Gentle Standard Chartered Blue-Green Pastel Background */
    .stApp {
        background: linear-gradient(135deg, #F5F9FC 0%, #F3F9F6 100%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Lock main container width so layout stays fixed regardless of window resizing */
    .block-container {
        width: 920px !important;
        max-width: 920px !important;
        min-width: 920px !important;
        margin: 0 auto !important;
        padding-top: 2.2rem !important;
        padding-bottom: 3rem !important;
    }

    /* Main Title */
    .scb-title {
        color: #0B3C5D;
        font-size: 1.65rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #D9E6F2;
        letter-spacing: -0.3px;
    }

    /* Section Headers */
    .section-header {
        color: #005A9C;
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 1.2rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #2EB872;
        padding-left: 10px;
    }

    /* Fixed-Size 3-Card Row (Total width: 280*3 + 16*2 = 872px) */
    .metrics-grid {
        display: flex;
        flex-direction: row;
        gap: 16px;
        width: 872px;
        margin-bottom: 14px;
    }

    .metric-box {
        width: 280px;
        min-width: 280px;
        max-width: 280px;
        height: 105px;
        min-height: 105px;
        max-height: 105px;
        background-color: #FFFFFF;
        border: 1px solid #D5E3EE;
        border-top: 3px solid #0072CE;
        border-radius: 6px;
        padding: 14px 18px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 2px 6px rgba(11, 60, 93, 0.04);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #627D98;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Identical font size across all 3 metric outputs */
    .metric-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #102A43;
        line-height: 1.3;
        word-wrap: break-word;
        overflow-wrap: break-word;
        overflow: hidden;
    }

    /* Fixed-Size Highlighted Action Decision Box (872px x 72px) */
    .decision-box {
        width: 872px;
        min-width: 872px;
        max-width: 872px;
        height: 72px;
        min-height: 72px;
        max-height: 72px;
        border-radius: 6px;
        padding: 0 20px;
        margin-bottom: 14px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .decision-critical {
        background-color: #FDF2F2;
        border: 1px solid #F5C6CB;
        border-left: 5px solid #C92A2A;
        color: #842029;
    }

    .decision-warning {
        background-color: #FFF9DB;
        border: 1px solid #FFEEBA;
        border-left: 5px solid #E67700;
        color: #664D03;
    }

    .decision-standard {
        background-color: #EBF5FB;
        border: 1px solid #B8DAFF;
        border-left: 5px solid #0072CE;
        color: #084298;
    }

    .decision-routine {
        background-color: #EBFBEE;
        border: 1px solid #C3E6CB;
        border-left: 5px solid #2B8A3E;
        color: #0F5132;
    }

    .box-caption {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.85;
        margin-bottom: 3px;
    }

    .box-content {
        font-size: 1.0rem;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Fixed-Size Transcribed Spoken Enquiry Box (872px x 72px) */
    .transcript-box {
        width: 872px;
        min-width: 872px;
        max-width: 872px;
        height: 72px;
        min-height: 72px;
        max-height: 72px;
        background-color: #FFFFFF;
        border: 1px solid #D5E3EE;
        border-left: 5px solid #2EB872;
        border-radius: 6px;
        padding: 0 20px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: #243B53;
    }

    /* Fixed width for audio recorder widget */
    div[data-testid="stAudioInput"] {
        width: 872px !important;
        max-width: 872px !important;
    }

    div[data-testid="stExpander"] {
        width: 872px !important;
        max-width: 872px !important;
        background-color: #FFFFFF;
        border: 1px solid #D5E3EE;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Brand Header & Main Title
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/0/0c/Standard_Chartered_%282021%29.svg",
    width=185,
)
st.markdown(
    '<div class="scb-title">Conversational Voice-Routing & Enquiry Triage Prototype</div>',
    unsafe_allow_html=True,
)

# ==============================================================================
# Domain Taxonomy & Priority Sets (77 Classes)
# ==============================================================================
CRITICAL_INTENTS = {
    "cancel_transfer",
    "card_payment_not_recognised",
    "card_swallowed",
    "cash_withdrawal_not_recognised",
    "compromised_card",
    "direct_debit_payment_not_recognised",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
}

STANDARD_INTENTS = {
    "activate_my_card",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_wrong_exchange_rate",
    "cash_withdrawal_charge",
    "contactless_not_working",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "extra_charge_on_statement",
    "failed_transfer",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "refund_not_showing_up",
    "request_refund",
    "reverted_card_payment",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_failed",
    "top_up_reverted",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_not_received_by_recipient",
    "unable_to_verify_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
}

ROUTINE_INTENTS = {
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "change_pin",
    "country_support",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "order_physical_card",
    "receiving_money",
    "supported_cards_and_currencies",
    "top_up_by_cash_or_cheque",
    "top_up_limits",
    "topping_up_by_card",
    "transfer_into_account",
    "transfer_timing",
    "verify_my_identity",
    "visa_or_mastercard",
    "why_verify_identity",
}

BANKING77_LABELS = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "refund_not_showing_up",
    "request_refund",
    "reverted_card_payment",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

ID2LABEL_DICT = {i: name for i, name in enumerate(BANKING77_LABELS)}


# ==============================================================================
# Model Loading with Resource Caching
# ==============================================================================
@st.cache_resource(show_spinner="Initializing Banking Models...")
def load_pipelines():
    device = 0 if torch.cuda.is_available() else -1
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=device,
    )
    intent_pipe = pipeline(
        "text-classification",
        model="Cktsai/scb-banking77-distilbert-test-3",
        device=device,
    )
    return asr_pipe, intent_pipe


asr_pipe, intent_pipe = load_pipelines()


# ==============================================================================
# Pipeline & Decision Functions
# ==============================================================================
def audio2text(audio_path: str) -> str:
    """Stage 1: Transcribe audio using soundfile."""
    speech_array, sampling_rate = sf.read(audio_path)
    if len(speech_array.shape) > 1:
        speech_array = speech_array.mean(axis=1)

    results = asr_pipe(
        {"raw": speech_array.astype(np.float32), "sampling_rate": sampling_rate},
        generate_kwargs={"language": "english", "task": "transcribe"},
    )
    return results["text"].strip()


def text2intent(transcript: str) -> Tuple[str, float]:
    """Stage 2: Classify transcript into Banking77 intent label with confidence score."""
    result = intent_pipe(transcript)[0]
    raw_label = result["label"]
    confidence_score = float(result["score"])

    if raw_label.startswith("LABEL_"):
        label_id = int(raw_label.replace("LABEL_", ""))
        intent_label = ID2LABEL_DICT.get(label_id, raw_label)
    elif raw_label.isdigit():
        label_id = int(raw_label)
        intent_label = ID2LABEL_DICT.get(label_id, raw_label)
    else:
        intent_label = raw_label

    return intent_label, confidence_score


def make_routing_decision(
    intent: str,
    confidence: float,
    high_threshold: float = 0.75,
    low_threshold: float = 0.45,
) -> Dict[str, str]:
    """Stage 3: Decision Engine based on Banking77 Intent, Priority, and Confidence."""
    if intent in CRITICAL_INTENTS:
        priority = "Critical"
    elif intent in STANDARD_INTENTS:
        priority = "Standard"
    elif intent in ROUTINE_INTENTS:
        priority = "Routine"
    else:
        priority = "Unknown"

    if confidence < low_threshold:
        decision = "REPEAT_ENQUIRY — Request Customer to Repeat Enquiry"
        style_class = "decision-warning"
    elif priority == "Critical":
        decision = "ESCALATE_HUMAN — Immediate Transfer to Fraud & Security Specialist"
        style_class = "decision-critical"
    elif confidence < high_threshold:
        decision = "ROUTE_HUMAN — Transfer to General Customer Service Representative"
        style_class = "decision-warning"
    elif priority == "Standard":
        decision = f"CONFIRM_THEN_ROUTE — Confirm '{intent}' and Route to Specialist Queue"
        style_class = "decision-standard"
    else:
        decision = f"AUTOMATED_SERVICE — Direct to Automated Voice Service ('{intent}')"
        style_class = "decision-routine"

    return {
        "Intent": intent,
        "Confidence": f"{confidence:.2%}",
        "Priority": priority,
        "Decision": decision,
        "_style_class": style_class,
    }


# ==============================================================================
# Sidebar - Presets & Model Architecture
# ==============================================================================
with st.sidebar:
    st.markdown("**Simulation Control**")
    demo_choice = st.selectbox(
        "Select Preset Scenario:",
        [
            "Live Microphone Input",
            "Preset 1: Stolen Card (Critical)",
            "Preset 2: ATM Cash Error (Standard)",
            "Preset 3: Exchange Rate (Routine)",
            "Preset 4: Unclear Speech (Low Confidence)",
        ],
    )
    st.divider()
    st.caption("System Specifications")
    st.caption("ASR Model: openai/whisper-tiny")
    st.caption("Classifier: scb-banking77-distilbert-test-3")
    st.caption("Thresholds: Low < 45% | High >= 75%")

# ==============================================================================
# Main Interface (Fixed-Size Containers, Auto-Trigger)
# ==============================================================================
st.markdown(
    '<div class="section-header">Input Customer Speech (Simulate Customer\'s inquiry in phone banking)</div>',
    unsafe_allow_html=True,
)

recorded_audio = st.audio_input("Record Voice Enquiry")

with st.expander("Alternative Input: Upload Audio File or Enter Text"):
    col_upload, col_text = st.columns([1, 1])
    with col_upload:
        uploaded_audio = st.file_uploader(
            "Upload Audio (.wav, .mp3, .flac)",
            type=["wav", "mp3", "flac", "ogg"],
        )
    with col_text:
        custom_text_fallback = st.text_area(
            "Direct Text Input:",
            value=(
                "I lost my credit card yesterday, please freeze my account immediately."
                if demo_choice == "Preset 1: Stolen Card (Critical)"
                else (
                    "I withdrew money at the ATM but the machine didn't dispense cash although my balance was deducted."
                    if demo_choice == "Preset 2: ATM Cash Error (Standard)"
                    else (
                        "What is the current foreign exchange rate for Japanese Yen?"
                        if demo_choice == "Preset 3: Exchange Rate (Routine)"
                        else (
                            "Um... yeah maybe something is wrong with my thing..."
                            if demo_choice == "Preset 4: Unclear Speech (Low Confidence)"
                            else ""
                        )
                    )
                )
            ),
            height=80,
        )

# Auto-trigger pipeline execution
transcript_output = ""
audio_source = recorded_audio or uploaded_audio

if audio_source is not None:
    with st.spinner("Processing voice enquiry..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_source.read())
            tmp_path = tmp.name
        try:
            transcript_output = audio2text(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
elif custom_text_fallback.strip():
    transcript_output = custom_text_fallback.strip()

# ==============================================================================
# Output Section: Analysis & Triage Decision (Fixed-Size Boxes)
# ==============================================================================
if transcript_output:
    st.markdown(
        '<div class="section-header">Analysis & Triage Decision</div>',
        unsafe_allow_html=True,
    )

    intent_predicted, conf_score = text2intent(transcript_output)
    triage = make_routing_decision(intent_predicted, conf_score)

    # Fixed-size 3-column metrics + Fixed-size Decision + Fixed-size Transcript
    st.markdown(
        f"""
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Predicted Intent</div>
                <div class="metric-value">{triage["Intent"]}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Confidence Score</div>
                <div class="metric-value">{triage["Confidence"]}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Priority Level</div>
                <div class="metric-value">{triage["Priority"]}</div>
            </div>
        </div>
        <div class="decision-box {triage["_style_class"]}">
            <div class="box-caption">Action Decision</div>
            <div class="box-content">{triage["Decision"]}</div>
        </div>
        <div class="transcript-box">
            <div class="box-caption" style="color: #627D98;">Transcribed Spoken Enquiry</div>
            <div class="box-content" style="font-weight: 500;">"{transcript_output}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
