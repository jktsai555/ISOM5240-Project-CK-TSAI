import os
import tempfile
from typing import Dict, Tuple
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import pipeline

# ==============================================================================
# Page Configuration & Gentle SCB Theme Styling (Blue & Green Pastel)
# ==============================================================================
st.set_page_config(
    page_title="SCB Conversational Phone Banking Triage",
    page_icon="🏦",
    layout="wide",
)

# Custom CSS for Standard Chartered Gentle Blue & Green Palette
st.markdown(
    """
    <style>
    /* Gentle background gradient with SCB soft blue and mint green */
    .stApp {
        background: linear-gradient(135deg, #F4F9FC 0%, #F2F9F5 100%);
    }
    
    /* Main Title Styling */
    .scb-title {
        color: #0B4F6C;
        font-size: 2.1rem;
        font-weight: 700;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }
    
    /* Section Headers */
    .section-header {
        color: #0072CE;
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        border-left: 5px solid #38D200;
        padding-left: 12px;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D5E8F2;
        border-top: 4px solid #0072CE;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 114, 206, 0.05);
    }

    /* Transcript Box at the bottom */
    .transcript-box {
        background-color: #EBF5FB;
        border-left: 4px solid #0072CE;
        color: #1B4F72;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 1.05rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 2 & 3. Standard Chartered Logo & Clean Title
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/0/0c/Standard_Chartered_%282021%29.svg",
    width=210,
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
@st.cache_resource(show_spinner="Loading Deep Learning Models...")
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
    """Stage 1: Transcribe audio using soundfile (No ffmpeg system dependency)."""
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
        decision = "REPEAT_ENQUIRY — Ask Customer to Repeat Question"
        badge_type = "warning"
    elif priority == "Critical":
        decision = "ESCALATE_HUMAN — Immediate Transfer to Human Fraud Specialist"
        badge_type = "error"
    elif confidence < high_threshold:
        decision = "ROUTE_HUMAN — Transfer to General Customer Service Agent"
        badge_type = "warning"
    elif priority == "Standard":
        decision = f"CONFIRM_THEN_ROUTE — Confirm '{intent}' & Route to Specialist Queue"
        badge_type = "info"
    else:
        decision = f"AUTOMATED_SERVICE — Direct to Self-Service Voice Menu ('{intent}')"
        badge_type = "success"

    return {
        "Intent": intent,
        "Confidence": f"{confidence:.2%}",
        "Priority": priority,
        "Decision": decision,
        "_badge_type": badge_type,
    }


# ==============================================================================
# Sidebar - Presets & Model Architecture
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Presets & Configuration")
    demo_choice = st.selectbox(
        "Quick Demo Scenarios:",
        [
            "None (Use Live Voice)",
            "Demo 1: Stolen Card (Critical)",
            "Demo 2: ATM Cash Not Dispensed (Standard)",
            "Demo 3: Exchange Rate Query (Routine)",
            "Demo 4: Mumbled / Low Confidence Speech",
        ],
    )
    st.divider()
    st.caption("Architecture Specifications:")
    st.caption("• Pipeline 1: `openai/whisper-tiny`")
    st.caption("• Pipeline 2: `Cktsai/scb-banking77-distilbert-test-3`")
    st.caption("• Thresholds: Low < 45% | High >= 75%")

# ==============================================================================
# Main Interactive Interface (Auto-Trigger without Button)
# ==============================================================================
st.markdown(
    '<div class="section-header">Input Customer Speech (Simulate Customer\'s inquiry in phone banking)</div>',
    unsafe_allow_html=True,
)

recorded_audio = st.audio_input("🎙️ Record Voice Enquiry")

with st.expander("📂 Optional: Upload Audio File or Enter Text Fallback"):
    col_upload, col_text = st.columns([1, 1])
    with col_upload:
        uploaded_audio = st.file_uploader(
            "Upload Audio (.wav, .mp3, .flac)",
            type=["wav", "mp3", "flac", "ogg"],
        )
    with col_text:
        custom_text_fallback = st.text_area(
            "Or type spoken text:",
            value=(
                "I lost my credit card yesterday, please freeze my account immediately."
                if demo_choice == "Demo 1: Stolen Card (Critical)"
                else (
                    "I withdrew money at the ATM but the machine didn't dispense cash although my balance was deducted."
                    if demo_choice == "Demo 2: ATM Cash Not Dispensed (Standard)"
                    else (
                        "What is the current foreign exchange rate for Japanese Yen?"
                        if demo_choice == "Demo 3: Exchange Rate Query (Routine)"
                        else (
                            "Um... yeah maybe something is wrong with my thing..."
                            if demo_choice == "Demo 4: Mumbled / Low Confidence Speech"
                            else ""
                        )
                    )
                )
            ),
            height=80,
        )

# Automatically process as soon as audio is recorded/uploaded or preset text is active
transcript_output = ""
audio_source = recorded_audio or uploaded_audio

if audio_source is not None:
    with st.spinner("Transcribing and analyzing voice enquiry..."):
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
# Output Section: Analysis & Triage Decision
# ==============================================================================
if transcript_output:
    st.divider()
    st.markdown(
        '<div class="section-header">Analysis & Triage Decision</div>',
        unsafe_allow_html=True,
    )

    intent_predicted, conf_score = text2intent(transcript_output)
    triage = make_routing_decision(intent_predicted, conf_score)

    # 7. Only 3 Core Metrics: Predicted Intent, Confidence Score, Priority Level
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Intent", triage["Intent"])
    m2.metric("Confidence Score", triage["Confidence"])
    m3.metric("Priority Level", triage["Priority"])

    # Highlighted Action Decision right below metrics
    badge = triage["_badge_type"]
    if badge == "error":
        st.error(f"🚨 **Action Decision:** {triage['Decision']}")
    elif badge == "warning":
        st.warning(f"⚠️ **Action Decision:** {triage['Decision']}")
    elif badge == "info":
        st.info(f"⚖️ **Action Decision:** {triage['Decision']}")
    else:
        st.success(f"✅ **Action Decision:** {triage['Decision']}")

    # 8. Transcribed Spoken Enquiry placed at the very bottom
    st.markdown(
        f'<div class="transcript-box">🎙️ <b>Transcribed Spoken Enquiry:</b> <i>"{transcript_output}"</i></div>',
        unsafe_allow_html=True,
    )
