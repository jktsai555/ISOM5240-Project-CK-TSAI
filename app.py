import streamlit as st

st.set_page_config(
    page_title="Microphone Recording Test",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Streamlit Microphone Test")
st.write("按下錄音按鈕，說完後停止錄音，接著可直接播放剛剛的錄音。")

recorded_audio = st.audio_input(
    "請簡單說明來電原因",
    sample_rate=16000,
)

if recorded_audio is not None:
    st.success("錄音完成，可以播放確認。")

    # st.audio_input 回傳 UploadedFile-like object。
    # 轉成 bytes 後交給 st.audio 播放，避免讀取游標狀態造成問題。
    audio_bytes = recorded_audio.getvalue()
    st.audio(audio_bytes, format="audio/wav")

    st.caption(f"錄音檔大小：{len(audio_bytes) / 1024:.1f} KB")

    st.download_button(
        label="下載錄音檔",
        data=audio_bytes,
        file_name="customer_enquiry.wav",
        mime="audio/wav",
    )
else:
    st.info("尚未錄音。請允許瀏覽器使用麥克風。")

st.divider()
st.caption("Test scope: microphone recording and playback only. No speech-to-text model is loaded yet.")
