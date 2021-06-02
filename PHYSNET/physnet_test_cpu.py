from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import numpy as np
import random
import math
from torchvision import transforms
import cv2
import os
import scipy.io
import torch.nn as nn
from torch.nn.modules.utils import _triple
import pdb
import torchvision
import torchvision.transforms as transforms
from torch import autograd

class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):  
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T, 128,128]
        x_visual = x
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)       # x [16, T, 64,64]
        
        x = self.ConvBlock2(x)		    # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)	    	# x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)      # x [32, T/2, 32,32]    Temporal halve
        
        x = self.ConvBlock4(x)		    # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)	    	# x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)      # x [64, T/4, 16,16]
        

        x = self.ConvBlock6(x)		    # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)	    	# x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)      # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)		    # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)		    # x [64, T/4, 8, 8]
        x = self.upsample(x)		    # x [64, T/2, 8, 8]
        x = self.upsample2(x)		    # x [64, T, 8, 8]
        
        
        x = self.poolspa(x)     # x [64, T, 1,1]    -->  groundtruth left and right - 7 
        x = self.ConvBlock10(x)    # x [1, T, 1,1]
        
        rPPG = x.view(-1,length)            
        

        return rPPG, x_visual, x_visual3232, x_visual1616



class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))

            #if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            #else:
            #    loss += 1 - torch.abs(pearson)
            
            loss += 1 - pearson
            
            
        loss = loss/preds.shape[0]
        return loss
criterion_Pearson = Neg_Pearson()




#MODEL INSTANTIATION
model = PhysNet_padding_Encoder_Decoder_MAX()
model.load_state_dict(torch.load('C:\\Users\\nabee\\Desktop\\Final Project DIP\\PHYSNET-50 epochs\\model-phys\\physnet.pt', map_location='cpu'))


#GENERATING TEST DATA
loss_val = []
mse_1 = []
mae_1 = []
l=[]
face_img=[]
target_l=[]
target_a=[]
NB = 'C:\\Users\\nabee\\Desktop\\project1\\physnet\\test'
g=os.listdir(NB)
print(len(g))
for i in range(0,len(g)):
    n = NB +"/" + g[i] + "/"
    u = n
    
    if os.path.isdir(n):
        li = os.listdir(n)
        
        for images in range(0,len(li)):
            if li[images] == 'rppg.mat':
                mat = scipy.io.loadmat(u+li[images])
                target_l.append(np.array(mat['gtTrace']))
                
            if '.jpg' in li[images]:
                mat = cv2.imread(u+li[images])
                mat = cv2.resize(mat,(128,128))
                l.append(np.array(mat))
                
face_img.append(l)
face_img = np.array(face_img)
face_img = face_img[0]
print(face_img.shape)

target_a.append(target_l)
target_a = np.array(target_a)
target = target_a[0]
label = np.array(target[0])
for i in range(1,target.shape[0]):
    h = np.array(target[i])
    label = np.append(label,h,axis=0)
print(label.shape)



#GENERATING DATA LOADER               
full_data = []
face_img = torch.tensor(face_img)
label_t = torch.tensor(label)
for i in range(0,len(label)):
    full_data.append([face_img[i],label_t[i]])

print(len(full_data))

trainloader = torch.utils.data.DataLoader(full_data, shuffle=True, batch_size=128, drop_last = True)
print((trainloader))


#TESTING
# model = model.cuda()

count = 0
for data in trainloader:
    print("Batch: " + str(count + 1))
    its,lbls = data
#     its, lbls = its.cuda(), lbls.cuda()
    c_image = its.view(1,3,128,128,128)
#     c_image = autograd.Variable(c_image.cuda())
    rppg_label = lbls.view(1,128)
#     rppg_label = autograd.Variable(rppg_label.cuda())
    rPPG, x_visual, x_visual3232, x_visual1616 = model(c_image.float())
    rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)                                  #normalize
    rppg_label = (rppg_label-torch.mean(rppg_label)) /torch.std(rppg_label)              # normalize
    loss_ecg = criterion_Pearson(rPPG, rppg_label)

        
    print("The Batch loss is: " + str(loss_ecg))
    
#     loss_num = loss.cpu().detach().numpy()
    loss_num = loss_ecg.detach().numpy()
    loss_val.append(loss_num)

    mean_se = nn.MSELoss()
    mean_se1 = mean_se(rPPG,rppg_label)
#     mean_se2 = mean_se1.cpu().detach().numpy()
    mean_se2 = mean_se1.detach().numpy()
    mse_1.append(mean_se2)

    mean_ae = nn.L1Loss()
    mean_ae1 = mean_ae(rPPG,rppg_label)
#     mean_ae2 = mean_ae1.cpu().detach().numpy()
    mean_ae2 = mean_ae1.detach().numpy()
    mae_1.append(mean_ae2)
    count = count + 1

    
print("Testing ended....")


#PLOTTING GRAPHS

plt.figure(1)
plt.plot(loss_val)
plt.title('Pearson Loss')
plt.ylabel('loss_ecg')
plt.xlabel('epoch')

plt.savefig('C:\\Users\\nabee\\Desktop\\project1\\physnet\\pearson_loss.png')

# mse plot
plt.figure(2)
plt.subplot(211)
plt.plot(mse_1)
plt.title('Mean Square Error')
plt.ylabel('Error')
plt.xlabel('epoch')

#mae plot
plt.subplot(212)
plt.plot(mae_1)
plt.title('Mean Absolute Error')
plt.ylabel('Error')
plt.xlabel('epoch')
plt.tight_layout()

plt.savefig('C:\\Users\\nabee\\Desktop\\project1\\physnet\\error_physnet.png')



