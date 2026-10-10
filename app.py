import os
import tempfile

import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Banking Voice ASR Test",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Banking Voice ASR Test")
st.write(
    "按下錄音按鈕，說完後停止錄音。你可以先播放錄音，再按下轉換按鈕產生英文逐字稿。"
)

@st.cache_resource(show_spinner="正在載入語音辨識模型...")
def load_asr_model():
    return pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=-1,
    )

recorded_audio = st.audio_input(
    "請用英文簡單說明來電原因",
    sample_rate=16000,
)

if recorded_audio is None:
    st.info("尚未錄音。請允許瀏覽器使用麥克風。")
else:
    audio_bytes = recorded_audio.getvalue()

    st.success("錄音完成。")
    st.subheader("1. 播放錄音")
    st.audio(audio_bytes, format="audio/wav")
    st.caption(f"錄音檔大小：{len(audio_bytes) / 1024:.1f} KB")

    st.download_button(
        label="下載 WAV 錄音檔",
        data=audio_bytes,
        file_name="customer_enquiry.wav",
        mime="audio/wav",
    )

    st.subheader("2. 語音轉文字")

    if st.button("開始轉換成文字", type="primary", use_container_width=True):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_path = temp_audio.name

            asr = load_asr_model()

            with st.spinner("正在辨識語音，第一次執行需下載模型，可能稍久..."):
                result = asr(temp_path)

            transcript = result.get("text", "").strip()

            if transcript:
                st.success("語音辨識完成。")
                st.text_area(
                    "Transcript",
                    value=transcript,
                    height=140,
                )
                st.session_state["latest_transcript"] = transcript
            else:
                st.warning("模型沒有辨識出文字，請重新錄音並說得更清楚。")

        except Exception as error:
            st.error(f"語音辨識失敗：{error}")
            st.info(
                "請確認網路可下載模型，並確認 requirements.txt 中的套件已完成安裝。"
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

if "latest_transcript" in st.session_state:
    st.divider()
    st.caption("最近一次辨識結果")
    st.code(st.session_state["latest_transcript"], language=None)

st.divider()
st.caption(
    "Current scope: microphone recording, playback, WAV download, and English speech-to-text using Whisper Tiny English."
)
