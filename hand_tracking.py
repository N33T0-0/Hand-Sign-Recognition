import cv2
import mediapipe as mp
import keras
import numpy as np

# Load Model in
classification_model = keras.models.load_model('nn.keras')

def processNClassify(X):
    pred = classification_model.predict(X)
    print(np.argmax(pred))


# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()

    h, w, c = frame.shape
    
    if not ret:
        continue
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    cropped_frame = frame
    
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
                    x_max = x + 60
                if x < x_min:
                    x_min = x - 60
                if y > y_max:
                    y_max = y + 60
                if y < y_min:
                    y_min = y - 60
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            position_ary = np.asarray(position)
            position_ary = position_ary.reshape(1,-1)

            # # Draw landmarks on the frame
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cropped_frame = frame[y_min:y_max,x_min:x_max]
            
            temp_img = cv2.resize(cropped_frame,(100,100))
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(temp_img,75,150)
            edges = np.asarray(edges)
            edges = edges.reshape(1,-1)

            frame_norm = temp_img.reshape(1,-1)
            frame_norm = frame_norm/255

            img_zip = np.hstack((frame_norm,position_ary,edges))

            
            if position_ary.shape[1] == 42:
                processNClassify(img_zip)

            # cv2.imshow('Cropped Img', cropped_frame)
    
    # Display the frame with hand landmarks
    cv2.imshow('Hand Recognition', frame)

    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
