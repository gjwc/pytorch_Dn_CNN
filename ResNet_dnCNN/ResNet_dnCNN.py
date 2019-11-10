# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 09:45:06 2019

@author: Administrator
"""


# coding: utf-8

# In[251]:


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

os.environ['CUDA_VISIBLE_DEVICES']='0, 1'


# In[252]:


#path='C:\\Users\\Administrator\\Desktop\\JPEGImages'
'''*************************此方法读取耗时较多************************'''
#定义读取图像地址的函数
def get_image_path(directory):
    return [x.path for x in os.scandir(directory) if x.name.endswith(".jpg")or x.name.endswith(".png")]

#定义一个可以返回图像numpy数组的函数
def load_images(image_paths):
    iter_all_images=(cv.resize(cv.imread(fn),(256,256)) for fn in image_paths)
    for i,image in enumerate(iter_all_images):
        if i==0:
            all_images=np.empty((len(image_paths),)+image.shape,dtype=image.dtype)
        all_images[i]=image#将图像的numpy数据存放到numpy数据当中
    return all_images

#最后定义一个可以根据batch_size大小读取固定图像数量的函数
#这个函数根据上一个得到的图像数据数组，从中挑选出batch_size个图像数据并返回
def get_training_data(images,batch_size):
    indices=np.random.randint(len(images),size=batch_size)
    for i ,index in enumerate(indices):
        image=images[index]
        image=random_transform(image,**random_transform_args)#随机的数据增强
        if i==0:
            images=np.empty((batch_size,)+image.shape,image.dtype)
        images[i]=image
    return images

#图像加噪
'''
sig=np.linspace(0,50,10)
def add_gaussian_noise(image_in):
    row,col,ch=image_in.shape
    mean=0
   
    np.random.shuffle(sig)
    i=np.random.randint(1,10)
    sigma=sig[i]
    
    gauss=np.random.normal(mean,25,(row,col,ch))
    gauss=gauss.reshape(row,col,ch)
 
    noisy=image_in+gauss

    #noisy=np.clip(noisy,0,255)
    #noisy=noisy.astype('uint8')
    return noisy,gauss
'''
def add_gaussian_noise(image_in, noise_sigma):
    
    temp_image = np.float64(np.copy(image_in))
 
    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma
    #print(noise)
 
    noisy_image = np.zeros(temp_image.shape, np.float64)
    noise_label = np.zeros(temp_image.shape, np.float64)
    image=np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
        
        noise_label[:,:,0]=noise
        noise_label[:,:,1]=noise
        noise_label[:,:,2]=noise
    '''
    noisy_image[noisy_image>255]=255
    noisy_image[noisy_image<0]=0
    
    noisy_image=noisy_image.astype(np.uint8)
    noise=noise.astype(np.uint8)
   
    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    #print(noisy_image)
    #cv2.imshow("11",noise_image)
    '''
    return noisy_image,noise_label


# In[253]:

ori_path="D:\\denoise\\pytorch_DnCNN\\data\\train\\original\\"
noise_path="D:\\denoise\\pytorch_DnCNN\\data\\train\\noisy\\"
image_name=np.array([x for x in os.listdir(ori_path)])#得到文件名
#image_name=sorted(noise_path)
#print(image_name)
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
        
train_data=MyDataSet(ori_path,transform=transforms.ToTensor())
        


# In[254]:


train_loder=DataLoader(dataset=train_data,batch_size=10,shuffle=True)
#print(train_loder)

'''
images,labels=next(iter(train_loder))
img=torchvision.utils.make_grid(images)
img=img.numpy()
img=img.transpose(1,2,0)
plt.imshow(img)
plt.show()

label=torchvision.utils.make_grid(labels)
label=label.numpy()
label=label.transpose(1,2,0)
plt.imshow(label)
plt.show()

noise=images-labels
img_1=torchvision.utils.make_grid(noise)
img_1=img_1.numpy()
img_1=img_1.transpose(1,2,0)
plt.imshow(img_1)
plt.show()

imgs=images-labels
img_1=torchvision.utils.make_grid(imgs)
img_1=img_1.numpy()
img_1=img_1.transpose(1,2,0)
plt.imshow(img_1)
plt.show()
'''


# In[255]:


#对一个批次的 数据进行预览
'''
noise_images,labels,re_imgs=next(iter(train_loder))
img=torchvision.utils.make_grid(noise_images)
img=img.numpy().transpose(1,2,0)
label=torchvision.utils.make_grid(labels)
label=label.numpy().transpose(1,2,0)
re_img=torchvision.utils.make_grid(re_imgs)
re_img=re_img.numpy().transpose(1,2,0)
plt.imshow(img)
plt.show()
plt.imshow(label)
plt.show()
img1=img-label
img1=img1.astype(np.uint8)
#img1=torchvision.utils.make_grid(img1)
#img1=img1.numpy().transpose(1,2,0)
plt.imshow(img1)
plt.show()
plt.imshow(re_img)
plt.show()
'''


# In[259]:

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


# In[ ]:


model=Model()
model.load_state_dict(torch.load('model_BF_para.pkl'))
model.cuda()
cost=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters())


# In[ ]:

loss_value=[]
step_list=[]
n_epochs=100
step=0
for epoch in range(n_epochs):
    running_loss=0.0
    running_correct=0
    print("Epoch {}/{}".format(epoch,n_epochs))
    print("-"*10)
    for data in train_loder:
        x_train,y_train=data
        x_train=torch.tensor(x_train, dtype=torch.float32)
        y_train=torch.tensor(y_train,dtype=torch.float32)
        x_train,y_train=Variable(x_train,requires_grad=False).cuda(),Variable(y_train,requires_grad=False).cuda()
        outputs=model(x_train)
        
        optimizer.zero_grad()
        
        loss=cost(outputs,y_train)
        step_list.append(step)
        loss_value.append(loss)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        '''
        if(step%100==0):
            print("loss:"+str(loss.item()))
        '''
        step+=1
        if(step%1000==0):
            torch.save(model.state_dict(),'model_BF_para.pkl')
    
    print("Loss is:{:.4f},Train Accuracy is :{:.4f}%".format(running_loss/len(train_data),100*running_correct/len(train_data)))
                                                                                    
            
torch.save(model.state_dict(),'model_BF_para.pkl')
np.save('loss_value',loss_value)
np.save('step_list',step_list)
