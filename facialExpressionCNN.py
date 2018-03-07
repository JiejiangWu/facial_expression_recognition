# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:02:15 2018

@author: wyj
"""

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle  as pk
import os,math
import random

## cnn
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size =5,stride =1,padding=2).cuda()
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size =5,stride =1,padding=1).cuda()
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size =5,stride =1,padding=2).cuda()
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2).cuda()
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2).cuda()
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2).cuda()
        
        self.fc1 = nn.Linear(5*5*64,1*1*2048).cuda()
        self.dp1 = nn.Dropout(p=0.4).cuda()
        self.fc2 = nn.Linear(1*1*2048,1*1*1024).cuda()
        self.dp2 = nn.Dropout(p=0.4).cuda()
        self.out = nn.Linear(1*1*1024,7).cuda()
        
        self.bn = nn.BatchNorm2d(1).cuda()
        self.bnConv1 = nn.BatchNorm2d(32).cuda()
        self.bnConv2 = nn.BatchNorm2d(32).cuda()
        self.bnConv3 = nn.BatchNorm2d(64).cuda()
        self.bnFc1 = nn.BatchNorm1d(2048).cuda()
        self.bnFc2 = nn.BatchNorm1d(1024).cuda()


    def forward(self, x):
        x=self.bn(x)
        
        x=F.relu(self.bnConv1((self.conv1(x))))
        x=self.pool1(x)
        
        x=F.relu(self.bnConv2((self.conv2(x))))
        x=self.pool2(x)
        
        x=F.relu(self.bnConv3((self.conv3(x))))
        x=self.pool3(x)
        
        x = x.view(-1,5*5*64)
        x=F.relu(self.bnFc1(self.fc1(x)))
        x=self.dp1(x)
        
        x=F.relu(self.bnFc2(self.fc2(x)))
        x=self.dp2(x)
        
        x=self.out(x)
        return x        


if __name__ == "__main__":
    file = open("fer2013.bat", 'rb')
    trnLabel,trnData,tstLabel,tstData = pk.load(file)
    file.close()

    net = Net()
    net = nn.DataParallel(net)
    
    optimizer = optim.Adam(net.parameters(), lr = 0.001,betas = [0.5,0.5])    
    criterion = nn.BCELoss(size_average=True)

    batch_size = 8

    samples = range(0,data.shape[0])

    d_f_loss = np.ones(50001)
    d_r_loss = np.ones(50001)
    g_f_loss = np.ones(50001)
    recon_loss = np.ones(100001)
#    Loss=np.ones(100001)
    DACC = np.ones(50001)
    Acc=np.ones(50001)
    MIoU=np.ones(50001)
    d_acc = 0
    
    for epoch in range(50001):  
        if epoch%2000 == 0 and epoch != 0:
            model_name = './model/net' + 'Seat_BCE_withADV_net{d}_64'.format(d = epoch) + '.pkl'
            torch.save(net.state_dict(), model_name)
            
            
        slice = random.sample(samples, batch_size)
        trianing_data =data[slice]
        trianing_data = np.reshape(trianing_data,(batch_size,1,64,64,64))