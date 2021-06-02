################################ NO CHANGE ##########################################

import argparse
import parser
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.layers import ZeroPadding3D, Dense, Activation,Conv3D,MaxPooling3D,AveragePooling3D,Flatten,Dropout
from tensorflow.python.keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.io

#CUDA
parser = argparse.ArgumentParser()
parser.add_argument('--Device', default = 'cuda:0', type = str)
parser.add_argument('--DataParallel', default = 0, type = int)

LENGTH_VIDEO = 60
IMAGE_WIDTH = 25
IMAGE_HEIGHT = 25
IMAGE_CHANNELS = 1

HEART_RATES = np.linspace(50, 128, 79)
print((HEART_RATES))
NB_CLASSES=len(HEART_RATES)
print(NB_CLASSES)
labels = np.zeros(NB_CLASSES)




######################################### NO CHANGE ###################################
# prepare labels and label categories
EPOCHS = 1000
CONTINUE_TRAINING = False
SAVE_ALL_MODELS = False
train_loss = []
val_loss = []
train_acc = []
val_acc = []
mse_1 = []
mae_1 =[]


# 1.  DEFINE OR LOAD MODEL / WEIGHTS

if (CONTINUE_TRAINING == False):
    init_batch_nb = 0
    
    model = Sequential() 

    model.add(Conv3D(filters=32, kernel_size=(58,20,20), input_shape=(LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(NB_CLASSES, activation='softmax'))

 

else:
    # load model
    model = model_from_json(open('model/model_conv3D.json').read())
    model.load_weights('model/weights_conv3D.h5')
   
    # load statistics
    dummy = np.loadtxt('model/statistics_loss_acc.txt')
    init_batch_nb = dummy.shape[0]
    train_loss = dummy[:,0].tolist()
    train_acc = dummy[:,1].tolist()
    val_loss = dummy[:,2].tolist()
    val_acc = dummy[:,3].tolist()

 

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','mean_squared_error','mean_absolute_error'])

# data = {}
print((LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

 
# 2.  GENERATE TRAINING DATA
l = []
face_img =[]
target_l =[]
target_a =[]
target =[]
NB = '/home/eng/s/sxs190123/3dcnn-img/train'

g = os.listdir(NB)
for i in range(0,len(g)):    
    n = NB +"/" + g[i] + "/"
    u = n

    if os.path.isdir(n):
        li = os.listdir(n)

        for images in range(0,len(li)):
            if li[images] == 'label.mat':
                mat = scipy.io.loadmat(u+li[images])
                target_l.append(np.array(mat['gtHR']))


            if '.jpg' in li[images]:
                mat = cv2.imread(u+li[images])
                mat = mat[:,:,1]
                mat = cv2.resize(mat,(25,25))
                l.append(np.array(mat))


target_a.append(target_l)
target_a = np.array(target_a)
print(target_a.shape)

face_img.append(l)
face_img = np.array(face_img)
face_img = face_img[0]
face_img = np.expand_dims(face_img,3)
print(face_img.shape)

target = target_a[0]
label = np.array(target[0])
for i in range(1,target.shape[0]):
    h = np.array(target[i])
    label = np.append(label,h,axis=0)
print(label.shape)
print(len(label)) 
target_a = np.zeros((len(label)))

for i in range(len(label)):
    if label[i] in HEART_RATES:
        target_a[i] = label[i] - 50

label_cat = np_utils.to_categorical(target_a,num_classes=79)
x_train = np.zeros(shape=(len(face_img),60,25,25,1))
y_train = np.zeros(shape=(len(face_img),79))


for i in range(len(face_img)):
    img = np.tile(face_img[i],(60,1,1))
    img = np.reshape(img,(60,25,25,1))
    x_train[i,:,:,:,:] = img
    y_train[i,:] = np.reshape(label_cat[i],(79))
print(x_train.shape)
print(y_train.shape)


# 3.  GENERATE VALIDATION DATA
l = []
face_img =[]
target_l =[]
target_a =[]
target =[]

NB = '/home/eng/s/sxs190123/3dcnn-img/validation'
g = os.listdir(NB)
for i in range(0,len(g)):    
    n = NB +"/" + g[i] + "/"
    u = n

    if os.path.isdir(n):
        li = os.listdir(n)

        for images in range(0,len(li)):
            if li[images] == 'label.mat':
                mat = scipy.io.loadmat(u+li[images])
                target_l.append(np.array(mat['gtHR']))


            if '.jpg' in li[images]:
                mat = cv2.imread(u+li[images])
                mat = mat[:,:,1]
                mat = cv2.resize(mat,(25,25))
                l.append(np.array(mat))


target_a.append(target_l)
target_a = np.array(target_a)
print(target_a.shape)

face_img.append(l)
face_img = np.array(face_img)
face_img = face_img[0]
face_img = np.expand_dims(face_img,3)
print(face_img.shape)


target = target_a[0]
label = np.array(target[0])
for i in range(1,target.shape[0]):
    h = np.array(target[i])
    label = np.append(label,h,axis=0)
print(label.shape)
print(len(label))                

target_a = np.zeros((len(label)))
for i in range(len(label)):
    if label[i] in HEART_RATES:
        target_a[i] = label[i] - 50

label_cat = np_utils.to_categorical(target_a,num_classes=79)

x_validation = np.zeros(shape=(len(face_img),60,25,25,1))
y_validation = np.zeros(shape=(len(face_img),79))


for i in range(len(face_img)):
    img = np.tile(face_img[i],(60,1,1))
    img = np.reshape(img,(60,25,25,1))
    x_validation[i,:,:,:,:] = img
    y_validation[i,:] = np.reshape(label_cat[i],(79))


print(x_validation.shape)
print(y_validation.shape) 


# 4.  TRAIN ON BATCH
batch_nb = 0

for i in range(0, EPOCHS):
        print(y_train.shape)
        history = model.train_on_batch(x_train,y_train)
        train_loss.append(history[0])
        train_acc.append(history[1])
        mse_1.append(history[2])
        mae_1.append(history[3])
        print(train_loss,train_acc)
        history = model.evaluate(x_validation, y_validation, verbose=2)

    # A. Save the model only if the accuracy is greater than before
        if (SAVE_ALL_MODELS==False):

            if (batch_nb > 0):
                f1 = open('model/statistics_loss_acc.txt', 'a')
                f2 = open('model/errors.txt', 'a')
                # save model and weights if val_acc is greater than before

                if (history[1] > np.max(val_acc)):
                    model.save_weights('model/weights_conv3D.h5', overwrite=True)   # save (trained) weights
                    print('A new model has been saved!\n')

            else:

                if not os.path.exists('model'):
                    os.makedirs('model')


                f1 = open('model/statistics_loss_acc.txt', 'w')
                f2 = open('model/errors.txt', 'w')
                model_json = model.to_json()
                open('model/model_conv3D.json', 'w').write(model_json)        # save model architecture

 

        # B. Save the model every iteration
        else:

            if (batch_nb > 0):
                f1 = open('model/statistics_loss_acc.txt', 'a')
                f2 = open('model/errors.txt', 'a')

            else:

                if not os.path.exists('model'):
                    os.makedirs('model')

                f1 = open('model/statistics_loss_acc.txt', 'w')
                f2 = open('model/errors.txt', 'w')
                model_json = model.to_json()
                open('model/model_conv3D.json', 'w').write(model_json)                       # save model architecture


            model.save_weights('model/weights_conv3D_%04d.h5' % batch_nb, overwrite=True)    # save (trained) weights

        val_loss.append(history[0])
        val_acc.append(history[1])
        print('training: ' + str(batch_nb + 1) + '/' + str(EPOCHS) + ' done')
        print('training: loss=' + str(train_loss[batch_nb]) + ' acc=' + str(train_acc[batch_nb]))
        print('validation: loss=' + str(val_loss[batch_nb]) + ' acc=' + str(val_acc[batch_nb]) + '\n')

        # save learning state informations
        f1.write(str(train_loss[batch_nb]) + '\t' + str(train_acc[batch_nb]) + '\t' + str(val_loss[batch_nb]) + '\t' + str(val_acc[batch_nb]) + '\n')
        f1.close()
        f2.write(str(mse_1[batch_nb]) + '\t' + str(mae_1[batch_nb]) + '\n')
        f2.close()

        batch_nb +=1
        c = 0


# 5.  PLOT
# plot history for accuracy
plt.figure(1)
plt.subplot(211)
plt.plot(train_acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')

# plot history for loss
plt.subplot(212)
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')

plt.tight_layout()
plt.show()
plt.savefig('graph_3dcnn_img.png')


plt.figure(2)

# mse plot
plt.subplot(211)
plt.plot(mse_1)
plt.title('Mean Square Error')
plt.ylabel('Error')
plt.xlabel('epoch')
#plt.legend(['training', 'validation'], loc='upper left')

#mae plot
plt.subplot(212)
plt.plot(mae_1)
plt.title('Mean Absolute Error')
plt.ylabel('Error')
plt.xlabel('epoch')
#plt.legend(['training', 'validation'], loc='upper left')


plt.tight_layout()
plt.show()
plt.savefig('error_3dcnn_img.png')


               
