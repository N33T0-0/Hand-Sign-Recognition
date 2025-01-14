import streamlit as st
import keras
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO

from camera_input_live import camera_input_live
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from turn import get_ice_servers

# Decode
target = ['a','b','c','e','i','m','o','s','t','u']

# Load Model in
classification_model = keras.models.load_model('nn.keras')
# classification_model = joblib.load('random_forest.joblib')

def processNClassify(X):
    pred = classification_model.predict(X)
    # print(target[np.argmax(pred)])
    # print(np.argmax(pred))
    return target[np.argmax(pred)]

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

def run(frame):
    # Convert frame byte to 
    img = cv2.imdecode(np.frombuffer(frame.read(), np.uint8), 1)
        
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = frame_rgb.shape
    
    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    
    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    cropped_frame = frame_rgb
    
    position = []
    edges = []

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw Bounding Box
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                position.append([x,y])
                if x > x_max:
                    x_max = x + 30
                if x < x_min:
                    x_min = x - 30
                if y > y_max:
                    y_max = y + 30
                if y < y_min:
                    y_min = y - 30
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            cv2.rectangle(frame_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            position_ary = np.asarray(position)
            position_ary = position_ary.reshape(1,-1)

            # # Draw landmarks on the frame
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cropped_frame = frame_rgb[y_min:y_max,x_min:x_max]
            
            temp_img = cv2.resize(cropped_frame,(100,100))
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(temp_img,75,150)
            edges = np.asarray(edges)
            edges = edges.reshape(1,-1)

            frame_norm = temp_img.reshape(1,-1)
            frame_norm = frame_norm/255

            img_zip = np.hstack((frame_norm,position_ary,edges))

            
            if position_ary.shape[1] == 42:
                pred = processNClassify(img_zip)
                cv2.rectangle(frame_rgb, (x_min, y_min - 20), (x_min + 20, y_min), (0,255,0), -1)
                cv2.putText(frame_rgb, pred, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    # convert Image...
    image_conv = Image.fromarray(frame_rgb)

    img_buffer = BytesIO()
    image_conv.save(img_buffer, format='PNG')  # Save the image to the BytesIO object
    img_buffer.seek(0)

    return img_buffer


st.title("Hand Sign Recognition")

# image = run(camera_input_live(debounce=2000))

# if image:
#     st.image(image)

# class DummyTransformer(VideoTransformerBase):
#     def transform(self, frame):
#         return frame  # Simply pass the frame without modifications

# webrtc_streamer(
#     key="hand-sign-recognition",
#     mode=WebRtcMode.SENDRECV,
#     video_processor_factory=DummyTransformer,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True,  # Ensures asynchronous handling
#     rtc_configuration={"iceServers": get_ice_servers()},
# )

print(get_ice_servers())