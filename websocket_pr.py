from typing import Union

import base64
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import json

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8501",
]

class Item(BaseModel):
    frame: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/process-frame")
def read_item(item:Item):
    # Send a response if needed
    encoded_data = item.split(',')[1]
    return ({'status': 'frame received'})

def base64_to_image(base64_str):
    """Convert Base64 string to OpenCV image."""
    img_data = base64.b64decode(base64_str)
    np_array = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

def image_to_base64(image):
    """Convert OpenCV image to Base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive JSON data containing the frame from the frontend
        data = await websocket.receive_text()
        frame_json = json.loads(data)
        base64_frame = frame_json['frame']

        # Decode the frame
        frame = base64_to_image(base64_frame)

        # Process the frame (e.g., convert to grayscale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Encode the processed frame back to Base64
        processed_base64 = image_to_base64(gray_frame)

        # Send the processed frame back to the frontend
        await websocket.send_text(json.dumps({"frame": processed_base64}))