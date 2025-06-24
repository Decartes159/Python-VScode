import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.title("Real-time Video Stream with Streamlit-WebRTC")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")

        # Perform a simple operation (vertical flip)
        img = img[::-1, :, :]

        # Convert back to a VideoFrame and return
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    rtc_configuration={  # This is needed for deployment on Streamlit Cloud
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)