import streamlit as st
from streamlit_webrtc import webrtc_streamer

import hand_recognition as hr

st.title("Hand Sign Recognition")

hr.run()

# webrtc_streamer(key="streamer")