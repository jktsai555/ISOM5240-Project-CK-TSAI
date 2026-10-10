import io

import numpy as np
import soundfile as sf
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Banking Voice ASR Test",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Banking Voice ASR Test")
st.write("錄音後可先播放，再按下按鈕產生英文逐字稿。")

@st.cache_resource(show_spinner="正在載入語音辨識模型...")
def load_asr_model():
    return pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=-1,
    )


def wav_bytes_to_numpy(audio_bytes: bytes):
    """Decode Streamlit WAV bytes without relying on ffmpeg."""
    waveform, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
        always_2d=False,
    )

    # Convert stereo/multi-channel audio to mono.
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    waveform = np.asarray(waveform, dtype=np.float32)
    return waveform, sample_rate


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

    st.download_button(
        label="下載 WAV 錄音檔",
        data=audio_bytes,
        file_name="customer_enquiry.wav",
        mime="audio/wav",
    )

    st.subheader("2. 語音轉文字")

    if st.button("開始轉換成文字", type="primary", use_container_width=True):
        try:
            waveform, sample_rate = wav_bytes_to_numpy(audio_bytes)

            st.caption(
                f"Audio samples: {len(waveform):,} | "
                f"Sample rate: {sample_rate:,} Hz | "
                f"Duration: {len(waveform) / sample_rate:.1f} sec"
            )

            asr = load_asr_model()

            with st.spinner("正在辨識語音，第一次執行需下載模型，可能稍久..."):
                result = asr(
                    {
                        "raw": waveform,
                        "sampling_rate": sample_rate,
                    }
                )

            transcript = result.get("text", "").strip()

            if transcript:
                st.success("語音辨識完成。")
                st.text_area("Transcript", value=transcript, height=140)
                st.session_state["latest_transcript"] = transcript
            else:
                st.warning("沒有辨識出文字，請重新錄音並說得更清楚。")

        except Exception as error:
            st.error(f"語音辨識失敗：{error}")
            st.info("建議先錄製 5 至 20 秒的英文語音，並確認錄音可以正常播放。")

if "latest_transcript" in st.session_state:
    st.divider()
    st.caption("最近一次辨識結果")
    st.code(st.session_state["latest_transcript"], language=None)

st.divider()
st.caption(
    "No-ffmpeg version: WAV bytes are decoded with soundfile and passed to Whisper as a NumPy waveform."
)
