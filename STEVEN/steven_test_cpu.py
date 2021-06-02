import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.io
import math
import torch.nn as nn
from torch.nn.modules.utils import _triple
import pdb
import torch
from torch import autograd





class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

		
        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/(kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        
        # self-definition
        #intermed_channels = int((in_channels+intermed_channels)/2)

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()   ##   nn.Tanh()   or   nn.ReLU(inplace=True)


        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))  
        x = self.temporal_conv(x)                      
        return x


class MixA_Module(nn.Module):
    """ Spatial-Skin attention module"""
    def __init__(self):
        super(MixA_Module,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.AVGpool = nn.AdaptiveAvgPool1d(1)
        self.MAXpool = nn.AdaptiveMaxPool1d(1)
    def forward(self,x , skin):
        """
            inputs :
                x : input feature maps( B X C X T x W X H)
                skin : skin confidence maps( B X T x W X H)
            returns :
                out : attention value
                spatial attention: W x H
        """
        m_batchsize, C, T ,W, H = x.size()
        B_C_TWH = x.view(m_batchsize,C,-1)
        B_TWH_C = x.view(m_batchsize,C,-1).permute(0,2,1)
        B_TWH_C_AVG =  torch.sigmoid(self.AVGpool(B_TWH_C)).view(m_batchsize,T,W,H)
        B_TWH_C_MAX =  torch.sigmoid(self.MAXpool(B_TWH_C)).view(m_batchsize,T,W,H)
        B_TWH_C_Fusion = B_TWH_C_AVG + B_TWH_C_MAX + skin
        Attention_weight = self.softmax(B_TWH_C_Fusion.view(m_batchsize,T,-1))
        Attention_weight = Attention_weight.view(m_batchsize,T,W,H)
        # mask1 mul
        output = x.clone()
        for i in range(C):
            output[:,i,:,:,:] = output[:,i,:,:,:].clone()*Attention_weight
        
        return output, Attention_weight    


# for open-source
# skin segmentation + PhysNet + MixA3232 + MixA1616part4
class rPPGNet(nn.Module):
    def __init__(self, frames=64):  
        super(rPPGNet, self).__init__()
        
        self.ConvSpa1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvSpa3 = nn.Sequential(
            SpatioTemporalConv(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa4 = nn.Sequential(
            SpatioTemporalConv(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.ConvSpa5 = nn.Sequential(
            SpatioTemporalConv(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa6 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa7 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa8 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa9 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
 
        self.ConvSpa10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.ConvSpa11 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.ConvPart1 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.ConvPart2 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.ConvPart3 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.ConvPart4 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
           
        
        self.AvgpoolSpa = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.AvgpoolSkin_down = nn.AvgPool2d((2,2), stride=2)
        self.AvgpoolSpaTem = nn.AvgPool3d((2, 2, 2), stride=2)
        
        self.ConvSpa = nn.Conv3d(3, 16, [1,3,3],stride=1, padding=[0,1,1])
        
        
        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))    # attention to this value 
        

        # skin_branch
        self.skin_main = nn.Sequential(
            nn.Conv3d(32, 16, [1,3,3], stride=1, padding=[0,1,1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, [1,3,3], stride=1, padding=[0,1,1]),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        
        self.skin_residual = nn.Sequential(
            nn.Conv3d(32, 8, [1,1,1], stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        
        self.skin_output = nn.Sequential(
            nn.Conv3d(8, 1, [1,3,3], stride=1, padding=[0,1,1]),
            nn.Sigmoid(),   ## binary 
        )
        
        self.MixA_Module = MixA_Module()
        
    def forward(self, x):	    	# x [3, 64, 128,128]
        x_visual = x
          
        x = self.ConvSpa1(x)     # x [3, 64, 128,128]
        x = self.AvgpoolSpa(x)       # x [16, 64, 64,64]
        
        x = self.ConvSpa3(x)		    # x [32, 64, 64,64]
        x_visual6464 = self.ConvSpa4(x)	    	# x [32, 64, 64,64]
        x = self.AvgpoolSpa(x_visual6464)      # x [32, 64, 32,32]
        
        
        ## branch 1: skin segmentation
        x_skin_main = self.skin_main(x_visual6464)    # x [8, 64, 64,64]
        x_skin_residual = self.skin_residual(x_visual6464)   # x [8, 64, 64,64]
        x_skin = self.skin_output(x_skin_main+x_skin_residual)    # x [1, 64, 64,64]
        x_skin = x_skin[:,0,:,:,:]    # x [74, 64,64]
        
        
        ## branch 2: rPPG
        x = self.ConvSpa5(x)		    # x [64, 64, 32,32]
        x_visual3232 = self.ConvSpa6(x)	    	# x [64, 64, 32,32]
        x = self.AvgpoolSpa(x_visual3232)      # x [64, 64, 16,16]
        
        x = self.ConvSpa7(x)		    # x [64, 64, 16,16]
        x = self.ConvSpa8(x)	    	# x [64, 64, 16,16]
        x_visual1616 = self.ConvSpa9(x)	    	# x [64, 64, 16,16]
        
        
        ## SkinA1_loss
        x_skin3232 = self.AvgpoolSkin_down(x_skin)          # x [64, 32,32]
        x_visual3232_SA1, Attention3232 = self.MixA_Module(x_visual3232, x_skin3232)
        x_visual3232_SA1 = self.poolspa(x_visual3232_SA1)     # x [64, 64, 1,1]    
        ecg_SA1  = self.ConvSpa10(x_visual3232_SA1).squeeze(1).squeeze(-1).squeeze(-1)
        
        
        ## SkinA2_loss
        x_skin1616 = self.AvgpoolSkin_down(x_skin3232)       # x [64, 16,16]
        x_visual1616_SA2, Attention1616 = self.MixA_Module(x_visual1616, x_skin1616)
        ## Global
        global_F = self.poolspa(x_visual1616_SA2)     # x [64, 64, 1,1]    
        ecg_global = self.ConvSpa11(global_F).squeeze(1).squeeze(-1).squeeze(-1)
        
        ## Local
        Part1 = x_visual1616_SA2[:,:,:,:8,:8]
        Part1 = self.poolspa(Part1)     # x [64, 64, 1,1]    
        ecg_part1 = self.ConvSpa11(Part1).squeeze(1).squeeze(-1).squeeze(-1)
        
        Part2 = x_visual1616_SA2[:,:,:,8:16,:8]
        Part2 = self.poolspa(Part2)     # x [64, 64, 1,1]    
        ecg_part2 = self.ConvPart2(Part2).squeeze(1).squeeze(-1).squeeze(-1)
        
        Part3 = x_visual1616_SA2[:,:,:,:8,8:16]
        Part3 = self.poolspa(Part3)     # x [64, 64, 1,1]    
        ecg_part3 = self.ConvPart3(Part3).squeeze(1).squeeze(-1).squeeze(-1)
        
        Part4 = x_visual1616_SA2[:,:,:,8:16,8:16]
        Part4 = self.poolspa(Part4)     # x [64, 64, 1,1]    
        ecg_part4 = self.ConvPart4(Part4).squeeze(1).squeeze(-1).squeeze(-1)
        
        

        return x_skin, ecg_SA1, ecg_global, ecg_part1, ecg_part2, ecg_part3, ecg_part4, x_visual6464, x_visual3232



###############################################################

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
#             pearson = (N*sum_xy - sum_x*sum_y)/((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2)))

            #if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            #else:
            #    loss += 1 - torch.abs(pearson)
            
            loss += 1 - pearson
            
            
        loss = loss/preds.shape[0]
        return loss



criterion_Binary = nn.BCELoss()  # binary segmentation
criterion_Pearson = Neg_Pearson()   # rPPG singal 




# MODEL INSTANTIATION
model = rPPGNet()
model.load_state_dict(torch.load('C:\\Users\\nabee\\Desktop\\Final Project DIP\\STEVEN-50 epochs\\model-stev\\steven_net.pt', map_location='cpu'))

# GENERATION OF TEST FACE IMAGES AND RPPG DATA
loss_val =[]
loss_bin =[]
mse_1 =[]
mae_1 =[]
mse_2 =[]
mae_2 =[]



l=[]
face_img=[]
target_l=[]
target_a=[]
target=[]
NB = 'C:\\Users\\nabee\\Desktop\\project1\\STEVEN\\data_test'
g = os.listdir(NB)
g.sort()
print(len(g))
for i in range(0,len(g)):
    n = NB +"\\" + g[i] + "\\"
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
               
                

# GENERATION OF TEST SKIN MAP
z = []
skin_label = []
MB = 'C:\\Users\\nabee\\Desktop\\project1\\STEVEN\\skin_data_test'
m = os.listdir(MB)
m.sort()
print(len(m))
for i in range(0,len(m)):
    n = MB +"/" + m[i] + "/"
    u = n
    if os.path.isdir(n):
        li = os.listdir(n)
        
        for images in range(0,len(li)):
            if '.jpg' in li[images]:
                mat = cv2.imread(u+li[images],2)
                ret, mat = cv2.threshold(mat,127,255,cv2.THRESH_BINARY)
                mat = cv2.resize(mat,(64,64))
                z.append(np.array(mat))
                
skin_label.append(z)
skin_label=np.array(skin_label)
skin_label = skin_label[0]
print(skin_label.shape)



#LOADING INTO DATALOADER
full_data = []
face_img = torch.tensor(face_img)
label_t = torch.tensor(label)
skin_label = torch.tensor(skin_label)
for i in range(0,len(label)):
    full_data.append([face_img[i],label_t[i],skin_label[i]])

print(len(full_data))
trainloader = torch.utils.data.DataLoader(full_data, shuffle=True, batch_size=64, drop_last = True)
print(trainloader)



#TESTING

# model = model.cuda()

count = 0
print('Testing is starting......')
for data in trainloader:
    its,lbls,sts = data
#     its, lbls, sts = its.cuda(), lbls.cuda(), sts.cuda()
    c_image = its.view(1,3,64,128,128)
#     c_image = autograd.Variable(c_image.cuda())
    rppg_label = lbls.view(1,64)
#     rppg_label = autograd.Variable(rppg_label.cuda())
    skin_image = sts.view(1,64,64,64)
#     skin_image = autograd.Variable(skin_image.cuda())
    
    rppg_label = rppg_label.float()
    skin_image = skin_image.float()
    skin_map, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232  = model(c_image.float())


    loss_binary = criterion_Binary(skin_map, skin_image)  

    rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize2
    rPPG_SA1 = (rPPG_SA1-torch.mean(rPPG_SA1)) /torch.std(rPPG_SA1)	 	# normalize2
    rPPG_SA2 = (rPPG_SA2-torch.mean(rPPG_SA2)) /torch.std(rPPG_SA2)	 	# normalize2
    rPPG_SA3 = (rPPG_SA3-torch.mean(rPPG_SA3)) /torch.std(rPPG_SA3)	 	# normalize2
    rPPG_SA4 = (rPPG_SA4-torch.mean(rPPG_SA4)) /torch.std(rPPG_SA4)	 	# normalize2
    rPPG_aux = (rPPG_aux-torch.mean(rPPG_aux)) /torch.std(rPPG_aux)	 	# normalize2

    loss_ecg = criterion_Pearson(rPPG, rppg_label)
    loss_ecg1 = criterion_Pearson(rPPG_SA1, rppg_label)
    loss_ecg2 = criterion_Pearson(rPPG_SA2, rppg_label)
    loss_ecg3 = criterion_Pearson(rPPG_SA3, rppg_label)
    loss_ecg4 = criterion_Pearson(rPPG_SA4, rppg_label)
    loss_ecg_aux = criterion_Pearson(rPPG_aux, rppg_label)
    
    loss = 0.1*loss_binary +  0.5*(loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg
    print('Batch:' + str(count + 1))
    print("The batch loss is: " + str(loss))

    #BINARY LOSS
#     loss_skin = loss_binary.cpu().detach().numpy()
    loss_skin = loss_binary.detach().numpy()
    loss_bin.append(loss_skin)
    #TOTAL LOSS
#     loss_num = loss.cpu().detach().numpy()
    loss_num = loss.detach().numpy()
    loss_val.append(loss_num)
    #RPPG MSE
    mean_se = nn.MSELoss()
    mean_se1 = mean_se(rPPG,rppg_label)
#     mean_se2 = mean_se1.cpu().detach().numpy()
    mean_se2 = mean_se1.detach().numpy()
    print("The rppg MSE is:" + str(mean_se2))
    mse_1.append(mean_se2)
    #RPPG MAE
    mean_ae = nn.L1Loss()
    mean_ae1 = mean_ae(rPPG,rppg_label)
#     mean_ae2 = mean_ae1.cpu().detach().numpy()
    mean_ae2 = mean_ae1.detach().numpy()
    print("The rppg MAE is:" + str(mean_ae2))
    mae_1.append(mean_ae2)
    #SKIN MSE
    s_mean_se = nn.MSELoss()
    s_mean_se1 = s_mean_se(skin_map, skin_image)
#     s_mean_se2 = s_mean_se1.cpu().detach().numpy()
    s_mean_se2 = s_mean_se1.detach().numpy()
    print("The skin label MSE is:" + str(s_mean_se2))
    mse_2.append(s_mean_se2)
    #SKIN MAE
    s_mean_ae = nn.L1Loss()
    s_mean_ae1 = s_mean_ae(skin_map, skin_image)
#     s_mean_ae2 = s_mean_ae1.cpu().detach().numpy()
    s_mean_ae2 = s_mean_ae1.detach().numpy()
    print("The skin label MAE is:" + str(s_mean_ae2))
    mae_2.append(s_mean_ae2)
    count = count + 1
    
print('Testing ended......')


#PLOTTING GRAPHS
#NEW PLOT
plt.figure(1)
#TOTAL LOSS
plt.plot(loss_val)
plt.title('Total Loss')
plt.ylabel('loss')
plt.xlabel('batch')
plt.tight_layout()
# plt.show()
plt.savefig('C:\\Users\\nabee\\Desktop\\Final Project DIP\\STEVEN-50 epochs\\test_loss_stevnet.png')

#BINARY LOSS
plt.figure(2)
plt.plot(loss_bin)
plt.title('Binary Loss')
plt.ylabel('loss')
plt.xlabel('batch')

plt.tight_layout()
# plt.show()
plt.savefig('C:\\Users\\nabee\\Desktop\\Final Project DIP\\STEVEN-50 epochs\\test_binary_loss_stevnet.png')


#NEW PLOT
plt.figure(3)
# mse plot
plt.subplot(211)
plt.plot(mse_1)
plt.title('Mean Square Error')
plt.ylabel('Error')
plt.xlabel('batch')

#mae plot
plt.subplot(212)
plt.plot(mae_1)
plt.title('Mean Absolute Error')
plt.ylabel('Error')
plt.xlabel('batch')

plt.tight_layout()
# plt.show()
plt.savefig('C:\\Users\\nabee\\Desktop\\Final Project DIP\\STEVEN-50 epochs\\test_rppg_error_stevnet.png')

#NEW PLOT
plt.figure(4)
# mse plot
plt.subplot(211)
plt.plot(mse_2)
plt.title('Mean Square Error')
plt.ylabel('Error')
plt.xlabel('batch')

#mae plot
plt.subplot(212)
plt.plot(mae_2)
plt.title('Mean Absolute Error')
plt.ylabel('Error')
plt.xlabel('batch')

plt.tight_layout()
# plt.show()
plt.savefig('C:\\Users\\nabee\\Desktop\\Final Project DIP\\STEVEN-50 epochs\\test_skin_error_stevnet.png')
