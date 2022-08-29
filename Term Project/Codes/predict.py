#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:52:20 2022

@author: sanketbhave
"""

import os, sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from random import randint
import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import models

class ImageDataSet(Dataset):
    def __init__(self, train, test, val):
        attributes = pd.read_csv(r"./Big Data/archive/list_attr_celeba.csv")
        partition_df = pd.read_csv(r"./Big Data/archive/list_eval_partition.csv")
        self.dataset = attributes.join(partition_df.set_index('image_id'), on='image_id')
        if train:
            self.dataset = self.dataset.loc[self.dataset['partition']==0]
            self.images = self.dataset['image_id']
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])
        elif test:
            self.dataset = self.dataset.loc[self.dataset['partition']==1]
            self.images = self.dataset['image_id']
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
        elif val:  
            self.dataset = self.dataset.loc[self.dataset['partition']==2]
            self.images = self.dataset['image_id']
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])
        self.len = len(self.dataset)
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        image = cv2.imread(r"./Big Data/archive/img_align_celeba/img_align_celeba/"+self.images.iloc[index])
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        atrributes = torch.from_numpy(np.array(self.dataset.iloc[index, 1:41], dtype=np.int32))

        
        return {
            'image': image,
            'attributes': atrributes
        }

def predict(rank, size):
    dataset = ImageDataSet(train=False, test=True, val=False)
    #dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    sampler = DistributedSampler(dataset, num_replicas=size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=sampler)
    
    inception_v3 = models.inception_v3(pretrained=False)
    inception_v3.aux_logits = False
    num_final_in = inception_v3.fc.in_features
    NUM_FEATURES = 40
    inception_v3.fc = nn.Linear(num_final_in, NUM_FEATURES)
    
    model=DDP(inception_v3)
    #model = inception_v3
    
    model.load_state_dict(torch.load(r"./Big Data/model.checkpoint"))
    
    model.eval()
    
    for i, data in enumerate(dataloader):
        #data, target = Variable(data), Variable(target)
        data, target = data['image'], data['attributes']
        prediction = model(data)
        print(prediction.size())

# model()
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'earth'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    
    #torch.manual_seed(40)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    setup(int(sys.argv[1]), int(sys.argv[2]))
    #train(int(sys.argv[1]), int(sys.argv[2]))
    predict(int(sys.argv[1]), int(sys.argv[2]))

