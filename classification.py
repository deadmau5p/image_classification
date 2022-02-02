import matplotlib.pyplot as plt 
import seaborn as sns
import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

labels = ["rugby", "soccer"]
img_size = 224

def load_data(path):
    data  = []
    for label in labels:
        full_url = os.path.join(path, label)
        class_num = labels.index(label)
        for img in os.listdir(full_url):
            try:
                img_arr = cv2.imread(os.path.join(full_url, img))
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_img, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)


train_data = load_data('./image_class/input/train')
test_data = load_data('./image_class/input/test')
