
import streamlit as st
from camera_input_live import camera_input_live
import cv2
from PIL import Image
from io import BytesIO
import numpy as np

st.title("My first Streamlit app")
st.write("Hello, world")


def callback(frame):

    img = cv2.imdecode(np.frombuffer(frame.read(), np.uint8), 1)

    img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

    # convert Image...
    image_conv = Image.fromarray(img)

    img_buffer = BytesIO()
    image_conv.save(img_buffer, format='PNG')  # Save the image to the BytesIO object
    img_buffer.seek(0)

    return img_buffer


image = callback(camera_input_live(debounce=300))

# # Convert Image
# image_conv = Image.open(image)

# print(image_conv)

# img_buffer = BytesIO()
# image_conv.save(img_buffer, format='PNG')  # Save the image to the BytesIO object
# img_buffer.seek(0)


if image:
    st.image(image)