import os
import tempfile
from typing import Dict, Tuple
import numpy as np
import streamlit as st
import torch
from transformers import pipeline

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="SCB Conversational Phone Banking Triage",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Standard Chartered Hong Kong")
st.subheader("Conversational Voice-Routing & Enquiry Triage Prototype")
st.markdown(
    """
This prototype demonstrates conversational voice-routing for phone banking, replacing complex multi-level 
keypad menus. Customers speak their banking requests naturally in English, and the system dynamically determines 
whether to route to automated self-service, request single voice confirmation, or escalate to human specialists.
"""
)

# ==============================================================================
# Domain Taxonomy & Priority Sets (Internal Alignment Appendix A)
# Total: 77 Classes (8 Critical, 38 Standard, 31 Routine)
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
@st.cache_resource(show_spinner="Loading Deep Learning Models into Memory...")
def load_pipelines():
  device = 0 if torch.cuda.is_available() else -1

  # Pipeline 1: Whisper Tiny ASR (~39MB)
  asr_pipe = pipeline(
      "automatic-speech-recognition", model="openai/whisper-tiny", device=device
  )

  # Pipeline 2: Custom Fine-tuned DistilBERT on Banking77 (~260MB)
  intent_pipe = pipeline(
      "text-classification",
      model="Cktsai/scb-banking77-distilbert-test",
      device=device,
  )

  return asr_pipe, intent_pipe


asr_pipe, intent_pipe = load_pipelines()


# ==============================================================================
# Pipeline & Decision Functions
# ==============================================================================
def audio2text(audio_path: str) -> str:
  """Stage 1: Transcribe customer spoken audio into English text using Whisper-tiny."""
  results = asr_pipe(
      audio_path, generate_kwargs={"language": "english", "task": "transcribe"}
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
    decision = "REPEAT_ENQUIRY"
    badge_type = "warning"
    action = (
        "Audio unclear / low confidence. System requests customer to repeat the"
        " enquiry."
    )
    queue = "Automated Clarification Prompt"
  elif priority == "Critical":
    decision = "ESCALATE_HUMAN"
    badge_type = "error"
    action = (
        "Immediate escalation to Human Fraud / Security Specialist. Bypass"
        " automated menus."
    )
    queue = "Emergency Card & Security Support"
  elif confidence < high_threshold:
    decision = "ROUTE_HUMAN"
    badge_type = "warning"
    action = f"Moderate confidence ({confidence:.1%}). Safely route to general customer service agent."
    queue = "General Customer Service Queue"
  elif priority == "Standard":
    decision = "CONFIRM_THEN_ROUTE"
    badge_type = "info"
    action = f"Read back detected need '{intent}' and ask customer for single confirmation before routing."
    queue = "Specialist Automated Routing Queue"
  else:
    decision = "AUTOMATED_SERVICE"
    badge_type = "success"
    action = f"High confidence routine enquiry. Direct customer into automated voice menu for '{intent}'."
    queue = "Self-Service Digital Voice Menu"

  return {
      "Intent": intent,
      "Confidence": f"{confidence:.2%}",
      "Priority": priority,
      "Decision": decision,
      "Recommended Action": action,
      "Assigned Queue": queue,
      "_badge_type": badge_type,
  }


# ==============================================================================
# Sidebar - Presets & Model Architecture
# ==============================================================================
with st.sidebar:
  st.header("⚙️ Presets & Configuration")
  st.markdown("Select a preset scenario if microphone is unavailable:")

  demo_choice = st.selectbox(
      "Preset Scenarios:",
      [
          "None (Use Live Voice / Custom Input)",
          "Demo 1: Stolen Card (Critical)",
          "Demo 2: ATM Cash Not Dispensed (Standard)",
          "Demo 3: Exchange Rate Query (Routine)",
          "Demo 4: Mumbled / Low Confidence Speech",
      ],
  )

  st.divider()
  st.caption("Architecture Specifications:")
  st.caption("• Pipeline 1: `openai/whisper-tiny` (ASR)")
  st.caption("• Pipeline 2: `Cktsai/scb-banking77-distilbert-test`")
  st.caption("• Decision Engine: Dual-Threshold (45% / 75%)")

# ==============================================================================
# Main Interactive Interface
# ==============================================================================
st.markdown("### Step 1: Input Customer Speech")
st.write(
    "Click the microphone button to record your banking enquiry directly from"
    " your browser:"
)

# 即時麥克風錄音輸入
recorded_audio = st.audio_input("🎙️ Record Voice Enquiry")

col_upload, col_text = st.columns([1, 1])

with col_upload:
  uploaded_audio = st.file_uploader(
      "Or upload an audio file (.wav, .mp3, .flac)",
      type=["wav", "mp3", "flac", "ogg"],
  )

with col_text:
  custom_text_fallback = st.text_area(
      "Or type/paste spoken enquiry text directly:",
      value=(
          "I lost my credit card yesterday, please freeze my account immediately."
          if demo_choice == "Demo 1: Stolen Card (Critical)"
          else (
              "I withdrew money at the ATM but the machine didn't dispense cash"
              " although my balance was deducted."
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
      height=90,
  )

# Trigger Button
transcript_output = ""

if st.button("🚀 Process Phone Banking Enquiry", type="primary"):
  with st.spinner("Processing Voice through Deep Learning Pipelines..."):
    audio_source = recorded_audio or uploaded_audio

    # Case A: Live Mic Recording or Uploaded Audio File
    if audio_source is not None:
      with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_source.read())
        tmp_path = tmp.name

      try:
        transcript_output = audio2text(tmp_path)
      finally:
        if os.path.exists(tmp_path):
          os.remove(tmp_path)

    # Case B: Direct Text / Preset
    elif custom_text_fallback.strip():
      transcript_output = custom_text_fallback.strip()
    else:
      st.error(
          "Please record voice via microphone, upload an audio file, or enter"
          " text."
      )

  # Step 2: Display Pipeline Analysis & Results
  if transcript_output:
    st.divider()
    st.markdown(
        "### Step 2: Multimodal Pipeline Analysis & Triage Decision"
    )

    intent_predicted, conf_score = text2intent(transcript_output)
    triage = make_routing_decision(intent_predicted, conf_score)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("1. ASR Transcript", f"{len(transcript_output.split())} Words")
    m2.metric("2. Predicted Intent", triage["Intent"])
    m3.metric("3. Confidence Score", triage["Confidence"])
    m4.metric("4. Priority Level", triage["Priority"])

    st.markdown(
        f"**Transcribed Spoken Enquiry:** *\"{transcript_output}\"*"
    )

    # Decision Banner
    badge = triage["_badge_type"]
    if badge == "error":
      st.error(f"🚨 **Action Decision: {triage['Decision']}**")
    elif badge == "warning":
      st.warning(f"⚠️ **Action Decision: {triage['Decision']}**")
    elif badge == "info":
      st.info(f"⚖️ **Action Decision: {triage['Decision']}**")
    else:
      st.success(f"✅ **Action Decision: {triage['Decision']}**")

    # Detailed Dispatch Summary Table
    st.markdown("#### 📋 Operational Routing Dispatch")
    st.table({
        "Attribute": [
            "Assigned Service Queue",
            "Recommended Operational Action",
            "Customer Experience Path",
        ],
        "Details": [
            triage["Assigned Queue"],
            triage["Recommended Action"],
            (
                "Immediate Escalation to Human Specialist"
                if triage["Priority"] == "Critical"
                else (
                    "Single Voice Confirmation -> Automated Routing"
                    if triage["Priority"] == "Standard"
                    else "Direct Self-Service Execution (Zero Keypad Menus)"
                )
            ),
        ],
    })
