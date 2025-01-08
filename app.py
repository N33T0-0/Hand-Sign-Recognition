import streamlit as st
from streamlit_webrtc import webrtc_streamer

import hand_recognition as hr

st.title("Hand Sign Recognition")

if st.button('Start Camera'):
    hr.run()

# webrtc_streamer(key="streamer")