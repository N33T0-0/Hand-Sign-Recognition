# import streamlit as st
# from camera_input_live import camera_input_live
# import cv2
# from PIL import Image
# from io import BytesIO
# import numpy as np
# import mediapipe as mp

# st.title("My first Streamlit app")
# st.write("Hello, world")

# # Initialize MediaPipe Hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# # Initialize MediaPipe Drawing module for drawing landmarks
# mp_drawing = mp.solutions.drawing_utils

# def callback(frame):

#     img = cv2.imdecode(np.frombuffer(frame.read(), np.uint8), 1)

#     # Convert the frame to RGB format
#     frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Process the frame to detect hands
#     results = hands.process(frame_rgb)
    
#     # Check if hands are detected
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw landmarks on the frame
#             mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # convert Image...
#     image_conv = Image.fromarray(frame_rgb)

#     img_buffer = BytesIO()
#     image_conv.save(img_buffer, format='PNG')  # Save the image to the BytesIO object
#     img_buffer.seek(0)

#     return img_buffer


# image = callback(camera_input_live())


# # image = camera_input_live(debounce=1000)


# if image:
#     st.image(image)


"""Video transforms with OpenCV"""

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from sample_utils.turn import get_ice_servers

_type = st.radio("Select transform type", ("noop", "cartoon", "edges", "rotate"))


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    if _type == "noop":
        pass
    elif _type == "cartoon":
        # prepare color
        img_color = cv2.pyrDown(cv2.pyrDown(img))
        for _ in range(6):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        img_color = cv2.pyrUp(cv2.pyrUp(img_color))

        # prepare edges
        img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.adaptiveThreshold(
            cv2.medianBlur(img_edges, 7),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2,
        )
        img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

        # combine color and edges
        img = cv2.bitwise_and(img_color, img_edges)
    elif _type == "edges":
        # perform edge detection
        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    elif _type == "rotate":
        # rotate image
        rows, cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="opencv-filter",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown(
    "This demo is based on "
    "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
    "Many thanks to the project."
)