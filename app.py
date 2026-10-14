import os
import tempfile
from typing import Dict, Tuple
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import pipeline

# ==============================================================================
# Page Configuration & Gentle SCB Styling (No ALL CAPS, Fixed Width, Unified Card)
# ==============================================================================
st.set_page_config(
    page_title="Standard Chartered | Voice Triage",
    layout="wide",
)

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #F5F9FC 0%, #F3F9F6 100%);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Fixed container width so resizing browser never distorts boxes */
.block-container {
    width: 900px !important;
    max-width: 900px !important;
    min-width: 900px !important;
    margin: 0 auto !important;
    padding-top: 2rem !important;
}

.scb-title {
    color: #0B3C5D;
    font-size: 1.55rem;
    font-weight: 600;
    margin-top: 0.4rem;
    margin-bottom: 1.4rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #D9E6F2;
}

.section-header {
    color: #0066B2;
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.6rem;
    border-left: 4px solid #38B2AC;
    padding-left: 10px;
}

/* Unified Fixed-Size Triage Card (860px wide) */
.triage-card {
    width: 860px;
    min-width: 860px;
    max-width: 860px;
    background-color: #FFFFFF;
    border: 1px solid #DCE8F2;
    border-radius: 10px;
    box-shadow: 0 4px 14px rgba(11, 60, 93, 0.04);
    overflow: hidden;
    margin-top: 0.5rem;
}

/* Top Row: 3 Equal Fixed Columns */
.metrics-row {
    display: flex;
    flex-direction: row;
    width: 860px;
    height: 92px;
    border-bottom: 1px solid #EDF2F7;
}

.metric-cell {
    width: 286px;
    min-width: 286px;
    max-width: 286px;
    padding: 16px 22px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border-right: 1px solid #EDF2F7;
}

.metric-cell:last-child {
    border-right: none;
}

.cell-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #64748B;
    margin-bottom: 4px;
}

.cell-value {
    font-size: 1.05rem;
    font-weight: 600;
    color: #0F2942;
    line-height: 1.25;
}

/* Middle Row: Highlighted Action Decision */
.decision-row {
    width: 860px;
    height: 68px;
    padding: 0 22px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border-bottom: 1px solid #EDF2F7;
}

.decision-critical {
    background-color: #FFF5F5;
    border-left: 5px solid #E53E3E;
    color: #9B2C2C;
}

.decision-warning {
    background-color: #FFFAF0;
    border-left: 5px solid #DD6B20;
    color: #9C4221;
}

.decision-standard {
    background-color: #F0F9FF;
    border-left: 5px solid #0072CE;
    color: #0C4A6E;
}

.decision-routine {
    background-color: #F0FDF4;
    border-left: 5px solid #2EB872;
    color: #14532D;
}

.decision-text {
    font-size: 1.02rem;
    font-weight: 600;
}

/* Bottom Row: Transcribed Spoken Enquiry */
.transcript-row {
    width: 860px;
    height: 64px;
    padding: 0 22px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: center;
    background-color: #FAFCFF;
}

.transcript-text {
    font-size: 0.96rem;
    color: #334E68;
    font-style: italic;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

div[data-testid="stAudioInput"],
div[data-testid="stExpander"] {
    width: 860px !important;
    max-width: 860px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Brand Header & Main Title
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/0/0c/Standard_Chartered_%282021%29.svg",
    width=175,
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
    speech_array, sampling_rate = sf.read(audio_path)
    if len(speech_array.shape) > 1:
        speech_array = speech_array.mean(axis=1)

    results = asr_pipe(
        {"raw": speech_array.astype(np.float32), "sampling_rate": sampling_rate},
        generate_kwargs={"language": "english", "task": "transcribe"},
    )
    return results["text"].strip()


def text2intent(transcript: str) -> Tuple[str, float]:
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
    if intent in CRITICAL_INTENTS:
        priority = "Critical"
    elif intent in STANDARD_INTENTS:
        priority = "Standard"
    elif intent in ROUTINE_INTENTS:
        priority = "Routine"
    else:
        priority = "Unknown"

    # Friendly, readable intent name (e.g., "Lost or stolen card")
    friendly_intent = intent.replace("_", " ").capitalize()

    if confidence < low_threshold:
        decision = "Please ask the customer to repeat their enquiry"
        style_class = "decision-warning"
    elif priority == "Critical":
        decision = "Immediate transfer to Fraud & Security Specialist"
        style_class = "decision-critical"
    elif confidence < high_threshold:
        decision = "Transfer to General Customer Service Representative"
        style_class = "decision-warning"
    elif priority == "Standard":
        decision = f"Confirm '{friendly_intent}' with customer and route to Specialist Queue"
        style_class = "decision-standard"
    else:
        decision = f"Direct to Automated Voice Service for '{friendly_intent}'"
        style_class = "decision-routine"

    return {
        "Intent": friendly_intent,
        "Confidence": f"{confidence:.1%}",
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
# Main Interface
# ==============================================================================
st.markdown(
    '<div class="section-header">Input Customer Speech</div>',
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
# Output Section: Analysis & Triage Decision (Single Unified Fixed Card)
# ==============================================================================
if transcript_output:
    st.markdown(
        '<div class="section-header">Analysis & Triage Decision</div>',
        unsafe_allow_html=True,
    )

    intent_predicted, conf_score = text2intent(transcript_output)
    triage = make_routing_decision(intent_predicted, conf_score)

    card_html = (
        f'<div class="triage-card">'
        f'<div class="metrics-row">'
        f'<div class="metric-cell"><div class="cell-label">Predicted Intent</div><div class="cell-value">{triage["Intent"]}</div></div>'
        f'<div class="metric-cell"><div class="cell-label">Confidence Score</div><div class="cell-value">{triage["Confidence"]}</div></div>'
        f'<div class="metric-cell"><div class="cell-label">Priority Level</div><div class="cell-value">{triage["Priority"]}</div></div>'
        f'</div>'
        f'<div class="decision-row {triage["_style_class"]}">'
        f'<div class="cell-label">Action Decision</div>'
        f'<div class="decision-text">{triage["Decision"]}</div>'
        f'</div>'
        f'<div class="transcript-row">'
        f'<div class="cell-label">Transcribed Spoken Enquiry</div>'
        f'<div class="transcript-text">"{transcript_output}"</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)
