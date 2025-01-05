# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
# import av

# st.title('Hand Sign Recognition')

# # enable = st.checkbox("Enable camera")
# # picture = st.camera_input("video a picture")


# import cv2
# import mediapipe as mp

# # Initialize MediaPipe Hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# # Initialize MediaPipe Drawing module for drawing landmarks
# mp_drawing = mp.solutions.drawing_utils

# # # Open a video capture object (0 for the default camera)
# # cap = cv2.VideoCapture(0)


# # Process Image
# def recognise(frame: av.VideoFrame):
#     print('in recognize')
#     img = frame.to_ndarray(format="rgb24")
#     st.image(img)
    
#     # # Convert the frame to RGB format
#     # frame_rgb = cv2.cvtColor(frame, img.COLOR_BGR2RGB)
    
#     # # Process the frame to detect hands
#     # results = hands.process(frame_rgb)
    
#     # # Check if hands are detected
#     # if results.multi_hand_landmarks:
#     #     for hand_landmarks in results.multi_hand_landmarks:
#     #         # Draw landmarks on the frame
#     #         mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
#     # Display the frame with hand landmarks
#     # display = cv2.imshow('Hand Recognition', img)

#     return av.VideoFrame.from_ndarray(img,format="rgb24")


# # # Release the video capture object and close the OpenCV windows
# # cap.release()
# # cv2.destroyAllWindows()

# webrtc_streamer(
#     key="streamer",
#     video_frame_callback=recognise,
#     sendback_audio=False
#     )

import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import numpy as np
import av

st.title("OpenCV Filters on Video Stream")

filter = "none"


def transform(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")

    if filter == "blur":
        img = cv2.GaussianBlur(img, (21, 21), 0)
    elif filter == "canny":
        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    elif filter == "grayscale":
        # We convert the image twice because the first conversion returns a 2D array.
        # the second conversion turns it back to a 3D array.
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif filter == "sepia":
        kernel = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        )
        img = cv2.transform(img, kernel)
    elif filter == "invert":
        img = cv2.bitwise_not(img)
    elif filter == "none":
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")


col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

with col1:
    if st.button("None"):
        filter = "none"
with col2:
    if st.button("Blur"):
        filter = "blur"
with col3:
    if st.button("Grayscale"):
        filter = "grayscale"
with col4:
    if st.button("Sepia"):
        filter = "sepia"
with col5:
    if st.button("Canny"):
        filter = "canny"
with col6:
    if st.button("Invert"):
        filter = "invert"


webrtc_streamer(
    key="streamer",
    video_frame_callback=transform,
    sendback_audio=False
    )