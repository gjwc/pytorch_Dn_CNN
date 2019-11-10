# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:34:23 2019

@author: Administrator
"""

import torch
import cv2 as cv
import os
import numpy as np
import torchvision.transforms
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
import os
ori_path="D:\\denoise\\pytorch_DnCNN\\data\\train\\original\\"
noise_path="D:\\denoise\\pytorch_DnCNN\\data\\train\\noisy\\"
image_name=np.array([x for x in os.listdir(ori_path)])#得到文件名
class MyDataSet(Dataset):
    def __init__(self,root,transform=None, target_transform=None):
        super(MyDataSet,self).__init__()
        #self.image_files=np.array([x.path for x in os.scandir(root) if x.name.endswith(".jpg")or x.name.endswith(".png")])
        self.image_files=np.array([x for x in os.listdir(root)])
        self.transform=transform
        self.target_transform=target_transform
        #print(self.image_files)
    
    def __getitem__(self,index):
        img=cv.imread(ori_path+self.image_files[index])
        noise=cv.imread(noise_path+self.image_files[index])
        re_img=cv.resize(img,(180,180),interpolation=cv.INTER_AREA)
        re_noise=cv.resize(noise,(180,180),interpolation=cv.INTER_AREA)
        noise=re_noise-re_img
        if self.transform is not None:
            re_img=self.transform(re_img)
            re_noise=self.transform(re_noise)
            noise=self.transform(noise)
        return re_noise,re_img
    
    def __len__(self):
        return len(self.image_files)
    
#卷积网络搭建
class Residual_Block(torch.nn.Module):
    def __init__(self,i_channel,o_channel,downsample=None):
        super(Residual_Block,self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=i_channel,out_channels=o_channel,kernel_size=3,stride=1,padding=1)
        self.bn1=torch.nn.BatchNorm2d(o_channel)
        self.relu1=torch.nn.ReLU(inplace=True)
        
        self.conv2=torch.nn.Conv2d(in_channels=o_channel,out_channels=o_channel,kernel_size=3,stride=1,padding=1)
        self.bn2=torch.nn.BatchNorm2d(o_channel)
        self.relu2=torch.nn.ReLU(inplace=True)
        
        self.conv3=torch.nn.Conv2d(in_channels=o_channel,out_channels=o_channel,kernel_size=3,stride=1,padding=1)
        self.bn3=torch.nn.BatchNorm2d(o_channel)
        self.relu3=torch.nn.ReLU(inplace=True)
        self.downsample=downsample
    
    def forward(self,x):
        #print("1111111111111111111111")
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu1(out)
        
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu2(out)
        
        out=self.conv3(out)
        out=self.bn3(out)
        if self.downsample:
            out+=residual
        out=self.relu3(out)
        return out
    
        

#卷积网络搭建
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv_1=torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True),
            torch.nn.ReLU()
            )
        self.block1=Residual_Block(64,64,True)
        self.block2=Residual_Block(64,64,True)
        self.block3=Residual_Block(64,64,True)
        self.block4=Residual_Block(64,64,True)
        self.block5=Residual_Block(64,64,True)
        self.block6=Residual_Block(64,64,True)
        self.block7=Residual_Block(64,64,True)
        self.conv2=torch.nn.Sequential(
            torch.nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1)
            )
        
     
    def forward(self,x):
        out=self.conv_1(x)
        out=self.block1(out)
        out=self.block2(out)
        out=self.block3(out)
        out=self.block4(out)
        out=self.block5(out)
        out=self.block6(out)
        out=self.block7(out)
        out=self.conv2(out)
        return x-out

        
train_data=MyDataSet(ori_path,transform=transforms.ToTensor())
        


# In[254]:


train_loder=DataLoader(dataset=train_data,batch_size=1,shuffle=True)

model=Model()
model.load_state_dict(torch.load('model_BF_para.pkl'))
model.eval() 

images,labels=next(iter(train_loder))
#噪声图像
img=torchvision.utils.make_grid(images)
img=img.numpy()
img=img.transpose(1,2,0)
plt.imshow(img)
plt.show()

#原始图像
label=torchvision.utils.make_grid(labels)
label=label.numpy()
label=label.transpose(1,2,0)
plt.imshow(label)

plt.show()
'''
#clean
output_1=model(images)
pre_1=images-labels
p1=torchvision.utils.make_grid(pre_1)
p1=p1.detach().numpy()
p1=p1.transpose(1,2,0)
#p=p
#p=p.astype(np.uint8)
plt.imshow(p1)
plt.show()
'''

output=model(images)
p2=torchvision.utils.make_grid(output)
p2=p2.detach().numpy()
p2=p2.transpose(1,2,0)
#p=p
#p=p.astype(np.uint8)
plt.imshow(p2)
plt.show()

'''
pre=images-output
p=torchvision.utils.make_grid(pre)
p=p.detach().numpy()
p=p.transpose(1,2,0)
#p=p
#p=p.astype(np.uint8)
plt.imshow(p)
plt.show()
'''

