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

# Get Path
base_path = 'Training_Data'

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

img_path, target = get_img_names_classes(base_path)
data = pd.DataFrame({
    'img_path' : img_path,
    'target'   : target
})

# Preprocessing
# Read And Resize Images
print('Resizing and GrayScaling Images...')
img_byte = []
for i in range(data.shape[0]):
    img = plt.imread(f'{base_path}/{data.iloc[i][0]}')
    img = cv2.resize(img,(224,224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_byte.append(img)
    # if i > 2:
    #     break
X = np.array(img_byte)
print('Complete!')

