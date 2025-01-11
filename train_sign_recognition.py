import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense,InputLayer

import joblib

# Get Path
base_path = 'training_data_2'

# Variables
img_byte = []
Y = []
P = []

def get_img_names_classes(base_path):
    img_path = []
    img_class = []
    for folder in os.listdir(base_path):
        dirname = os.path.join(base_path, folder)
        img_dir = os.listdir(dirname)
        n = len(img_dir)

        img_path += [os.path.join(folder, img) for img in img_dir]
        img_class += [folder] * n

    return img_path, img_class

def transform_img(img,target,path):
    rows, cols, c = img.shape
    #Vertical Flip
    img_flip = cv2.flip(img, 1)
    # Define points for perspective transformation (tilt up)
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    pts2 = np.float32([[cols*0.1, rows*0.1], [cols*0.9, rows*0.1], [0, rows], [cols, rows]])
    pts3 = np.float32([[0, 0], [cols, 0], [cols*0.1, rows*0.9], [cols*0.9, rows*0.9]])
    pts4 = np.float32([[0, 0], [cols*0.9, rows*0.1], [0, rows], [cols*0.9, rows*0.9]])
    pts5 = np.float32([[cols*0.1, rows*0.1], [cols, 0], [cols*0.1, rows*0.9], [cols, rows]])

    M_up = cv2.getPerspectiveTransform(pts1, pts2)
    M_down = cv2.getPerspectiveTransform(pts1, pts3)
    M_left = cv2.getPerspectiveTransform(pts1, pts4)
    M_right = cv2.getPerspectiveTransform(pts1, pts5)

    # Append Transformed Image
    img_byte.append(img)
    img_byte.append(cv2.warpPerspective(img, M_up, (int(cols), int(rows))))
    img_byte.append(cv2.warpPerspective(img, M_down, (int(cols), int(rows))))
    img_byte.append(cv2.warpPerspective(img, M_left, (int(cols), int(rows))))
    img_byte.append(cv2.warpPerspective(img, M_right, (int(cols), int(rows))))

    img_byte.append(img_flip)
    img_byte.append(cv2.warpPerspective(img_flip, M_up, (int(cols), int(rows))))
    img_byte.append(cv2.warpPerspective(img_flip, M_down, (int(cols), int(rows))))
    img_byte.append(cv2.warpPerspective(img_flip, M_left, (int(cols), int(rows))))
    img_byte.append(cv2.warpPerspective(img_flip, M_right, (int(cols), int(rows))))

    for i in range(0,10):
        Y.append(target)
        P.append(path)

img_path, target = get_img_names_classes(base_path)
data = pd.DataFrame({
    'img_path' : img_path,
    'target'   : target
})

# Preprocessing
# Read And Resize Images
print('Resizing and GrayScaling Images...')
for i in range(data.shape[0]):
    img = plt.imread(f'{base_path}/{data.iloc[i][0]}')
    img = cv2.resize(img,(550,380))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_byte.append(img)
    transform_img(img,data['target'][i],data['img_path'][i])
    # if i > 2:
    #     break
X = np.array(img_byte)
print('Complete!')

# Detecting Hand
# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils


print('Find, Crop, Resize and GrayScale Hands...')
# Find Hands
cropped_X = []
landmarks = []
for index, img in enumerate(X):
    h, w , c = img.shape
    results = hands.process(img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw Bounding Box
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            positions = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                positions.append([x,y])
                if x > x_max:
                    x_max = x + 30
                if x < x_min:
                    x_min = x - 30
                if y > y_max:
                    y_max = y + 30
                if y < y_min:
                    y_min = y -30
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0

            temp_img = img[y_min:y_max,x_min:x_max]

            temp_img = cv2.resize(temp_img,(100,100))
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

            cropped_X.append(temp_img)
            landmarks.append(positions)
            break
    else:
        print(index)

print('Completed!')

cropped_X = np.asarray(cropped_X)
landmarks = np.asarray(landmarks)
landmarks = landmarks.reshape(landmarks.shape[0],-1)

print(X.shape)
print(cropped_X.shape)
print(landmarks.shape)

# # Features


# # Normalize
# cropped_X = cropped_X.reshape(cropped_X.shape[0],-1)
# cropped_X = cropped_X/255

# print(X.shape)
# print(cropped_X.shape)
# print(landmarks.shape)

# # Join All Features
# X_train_zip = np.hstack((cropped_X,landmarks))

# print(X_train_zip.shape)
# print(X_train_zip[0])

# # Training
# Y = data['target']
# Y = LabelEncoder().fit_transform(Y)

# print('Seperating Data to Train and Test...')
# X_train, X_test, y_train, y_test = train_test_split(X_train_zip, Y, test_size = 0.2,stratify=Y, random_state = 42)
# print('Complete!')

# # Model
# Model = Sequential()

# input_neurons = X_train.shape[1]
# output_neuron = 10

# print(X_train.shape[1])

# nos_of_hidden_layer = 2
# neuron_hidden_layer_1 = 128
# neuron_hidden_layer_2 = 50

# Model.add(InputLayer(input_shape=(input_neurons,)))
# Model.add(Dense(units=neuron_hidden_layer_1,activation='relu'))
# Model.add(Dense(units=neuron_hidden_layer_2,activation='relu'))
# Model.add(Dense(units=output_neuron,activation='softmax'))

# print('Compiling Model...')

# Model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# print('Complete!')

# Model.summary()

# print('DataTypes')
# print(X_train.shape, X_train.dtype)
# print(X_test.shape, X_test.dtype)
# print(y_train.shape, y_train.dtype)
# print(y_test.shape, y_test.dtype)

# history = Model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 16, epochs = 10,verbose=2)

# y_pred = Model.predict(X_test)


# cm = confusion_matrix(y_test, y_pred)
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix: \n", cm)
# print("\nclassification_report: \n", classification_report(y_test, y_pred))

# # Summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Models Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epochs - Nos. of rounds')
# plt.legend(['Train', 'Validation'], loc='upper left', bbox_to_anchor = (1,1))
# plt.show()

# # Summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Loss')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs - Nos. of rounds')
# plt.legend(['Train', 'Validation'], loc='upper left', bbox_to_anchor = (1,1))
# plt.show()

# # # Take random Image
# # test_image = plt.imread("test_e.jpg")

# # print(test_image.shape)

# # y_pred = Model.predict_on_batch(test_image)
# # y_pred = (y_pred > 0.5)

# # plt.figure
# # plt.imshow(plt.imread("test_e.jpg"))
# # plt.show()
# # print(f'Class? \n(Predict): {y_pred}')

# # print(f"(Actual): e")


# # Dump Model to pkl file
# joblib.dump(Model, 'nn.pkl')