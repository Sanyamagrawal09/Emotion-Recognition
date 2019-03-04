# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 01:05:14 2019

@author: user
"""

import keras
import pandas as pd
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,merge,BatchNormalization
from keras.optimizers import *
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D,MaxPooling2D,AveragePooling2D
from keras.models import Sequential, Model
from keras import backend as k
import cv2
num_features = 64
num_labels = 7
batch_size = 64
i= 1
width, height = 48, 48
data_set=pd.read_csv('C:/Users/user/Downloads/deep learning/training_set/train/fer2013.csv')
pixels=data_set['pixels'].tolist()
emotions = pd.get_dummies(data_set['emotion']).as_matrix()
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')] # 2
    face = np.asarray(face).reshape(width, height)
    faces.append(face.astype('float32'))
    i=i+1
i=1
#adding gaussian noise to the dataset #preprocessing
for i in range(len(faces)):
    blur = cv2.GaussianBlur(faces[i-1], (5, 5), 0)
    faces.append(blur.astype('float32'))
emotions=np.vstack((emotions,emotions))
faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)
model = Sequential()

model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.0001,decay=1e-6),metrics=['accuracy'])
model.summary()
datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
datagen.fit(X_train)
validdatagen=ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
validdatagen.fit(X_val)
model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),steps_per_epoch=int(X_train.shape[0]/batch_size),epochs=60,validation_data=validdatagen.flow(X_val,y_val,batch_size=batch_size),validation_steps=int(X_val.shape[0]/batch_size))

#real time emotion detection
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
#for identifying the face in real time 
facecascade = cv2.CascadeClassifier('C:/Users/user/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret, colouredface = cap.read()
    gray = cv2.cvtColor(colouredface, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
         cv2.rectangle(colouredface, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
         roi_gray = gray[y:y + h, x:x + w]
         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
         prediction = model.predict(cropped_img)
         maxindex = int(np.argmax(prediction))
         cv2.putText(colouredface, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video', cv2.resize(colouredface,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()