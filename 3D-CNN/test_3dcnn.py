## UNCOMMENTING THESE TWO LINES WILL FORCE KERAS/TF TO RUN ON CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils import np_utils

import numpy as np
import scipy.io
import scipy.stats as sp
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import cv2
import os
from copy import copy



NB = 'C:\\Users\\nabee\\Desktop\\project1\\3dcnn-img\\testing'
g = os.listdir(NB)


IMAGE_WIDTH = 25
IMAGE_HEIGHT = 25
IMAGE_CHANNELS = 3
print(IMAGE_CHANNELS)

# 1. LOAD PRETRAINED MODEL
model = model_from_json(open('C:\\Users\\nabee\\Desktop\\Final Project DIP\\3dcnn-img\\model\\model_conv3D.json').read())
model.load_weights('C:\\Users\\nabee\\Desktop\\Final Project DIP\\3dcnn-img\\model\\weights_conv3D.h5')
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','mean_squared_error','mean_absolute_error'])


# 2. LOAD DATA
test_labels=[]
face_img = []

for i in range(len(g)):
    n = NB +"\\" + g[i] + "\\"
    u = n
    
    if os.path.isdir(n):
        li = os.listdir(n)
        
        for images in range(0,len(li)):
            if ".jpg" in li[images]:
                temp = cv2.imread(u + li[images], cv2.IMREAD_ANYCOLOR)
                temp = cv2.resize(temp,(25,25))
                if (IMAGE_CHANNELS==3):
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)/255
                    temp = temp[:,:,1]      # only the G component is currently used
                else:
                    temp = temp / 255

                face_img.append(temp)
            if "label.mat" in li[images]:
                mat = scipy.io.loadmat(u + li[images])
                test_labels.append(np.array(mat['gtHR']))
            


face_img = np.array(face_img)
face_img = np.expand_dims(face_img,3)
print(face_img.shape)


test_label = np.array(test_labels)
final = test_label[0]
for i in range(1,len(test_label)):
    h = np.array(test_label[i])
    final = np.append(final,h)
final_label = np.expand_dims(final,1)
print(final_label.shape)



HEART_RATES = np.linspace(50, 128, 79)
print((HEART_RATES))

target_a = np.zeros((len(final_label)))
for i in range(len(final_label)):
     if final_label[i] in HEART_RATES:
            target_a[i] = final_label[i] - 50
label_cat = np_utils.to_categorical(target_a,num_classes=79)
label_cat = np.expand_dims(label_cat,1)
print(label_cat.shape)

#TESTING
test_loss= []
test_acc=[]
mse = []
mae = []
for j in range(int(face_img.shape[0])):            #  NUMBER OF IMAGES  #
            xtest = face_img[j]
            xtest = np.tile(xtest, (60,1,1,1))
            xtest = np.expand_dims(xtest,0)
            h = model.evaluate(xtest,label_cat[j])
            test_loss.append(h[0])
            test_acc.append(h[1])
            mse.append(h[2])
            mae.append(h[3])


plt.figure(1)
plt.subplot(211)
plt.plot(test_acc)
plt.title('Testing Accuracy')
plt.ylabel('accuracy')
plt.xlabel('frame')


# plot history for loss
plt.subplot(212)
plt.plot(test_loss)
plt.title('Testing Loss')
plt.ylabel('loss')
plt.xlabel('frame')


plt.tight_layout()
plt.savefig('C:\\Users\\nabee\\Desktop\\Final Project DIP\\3dcnn-img\\test_graph_3dcnn.png')


plt.figure(2)

# mse plot
plt.subplot(211)
plt.plot(mse)
plt.title('Mean Square Error')
plt.ylabel('Error')
plt.xlabel('epoch')
#plt.legend(['training', 'validation'], loc='upper left')

#mae plot
plt.subplot(212)
plt.plot(mae)
plt.title('Mean Absolute Error')
plt.ylabel('Error')
plt.xlabel('epoch')
#plt.legend(['training', 'validation'], loc='upper left')


plt.tight_layout()
plt.savefig('C:\\Users\\nabee\\Desktop\\Final Project DIP\\3dcnn-img\\test_error_3dcnn.png')
