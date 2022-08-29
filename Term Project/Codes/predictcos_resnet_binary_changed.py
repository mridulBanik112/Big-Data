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
        attributes = pd.read_csv(r"./list_attr_celeba.csv")
        attributes = attributes.replace(-1, 0)
        partition_df = pd.read_csv(r"./list_eval_partition.csv")
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
        image = cv2.imread(r"./img_align_celeba/img_align_celeba/"+self.images.iloc[index])
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        atrributes = torch.from_numpy(np.array(self.dataset.iloc[index, 1:41], dtype=np.int32))
        image_id = self.dataset.iloc[index, 0:1].tolist()

        
        return {
            'image': image,
            'attributes': atrributes,
            'image_id':image_id
        }

def predict(rank, size):
    desired_image = '162775.jpg'
    n = 10
    output_dict  = {}

    # attributes1 = pd.read_csv(r"./list_attr_celeba.csv")
    # attributes1 = attributes1.replace(-1, 0)
    # partition_df1 = pd.read_csv(r"./list_eval_partition.csv")
    # dataset1 = attributes1.join(partition_df1.set_index('image_id'), on='image_id')
    # dataset1 = dataset1.loc[dataset1['partition']==1]
    # images = dataset1['image_id']
    # print("printing location")
    # print(dataset1[dataset1['image_id'] == '162773.jpg'].index.values)
    # #print(dataset1['image_id']==16773)
    # #print(len(images))
    # _id = '162773.jpg'
    # for i,j in enumerate(images):
    #     if i < 5:
    #         print(j)
    #         print(type(j))
    #         if j == _id:
    #             print(j)
    #             print(i)


    check = int(desired_image.split(".")[0])-162771
    dataset = ImageDataSet(train=False, test=True, val=False)
    #dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    sampler = DistributedSampler(dataset, num_replicas=size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler)

    # train dataloading
    train_dataset = ImageDataSet(train=True, test=False, val=False)
    #dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=size, rank=rank, shuffle=False, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, sampler=train_sampler)
    
    #inception_v3 = models.inception_v3(pretrained=False)
    inception_v3 = models.resnet18(pretrained=False)
    inception_v3.aux_logits = False
    num_final_in = inception_v3.fc.in_features
    NUM_FEATURES = 40
    inception_v3.fc = nn.Sequential(nn.Linear(num_final_in, NUM_FEATURES), nn.Sigmoid())
    
    model=DDP(inception_v3)
    #model = inception_v3
    checkpoint = torch.load(r"./model.checkpoint")
    #model.load_state_dict(torch.load(r"./model.checkpoint"))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    losses = torch.zeros(len(dataloader), 40)
    #file = open("similarity_log.txt","w")
    for i, data in enumerate(dataloader):
        if i == check:

        #data, target = Variable(data), Variable(target)
            data, target, test_image_id = data['image'], data['attributes'], data['image_id']
            print(len(test_image_id))
            print(test_image_id)
            prediction = model(data)
            print(prediction)
            # losses[i][:] = abs(target-prediction)
            # print(i)
            # value 0 and 1
            #prediction = (prediction>0.5).int()
            #print("prediction",prediction)
            #print("prediction size",prediction.size())
            #print("type attr",type(target))
            #print("label",target)
            #print("label size",target.size())
            for pred_val in prediction:
                for train_i, train_data in enumerate(train_dataloader):
                    tr_data, tr_target,tr_image_id = train_data['image'], train_data['attributes'],train_data['image_id']
                    #print("tr_image_id",len(tr_image_id[0]),tr_image_id[0][0])
                    for train_attr,train_id in zip(tr_target,tr_image_id[0]):

                        cos = torch.nn.CosineSimilarity(dim=0)
                        output = cos(pred_val, train_attr)
                        #print("="*30)
                        if output>=0.85:

                            print("Cosine Similarity of ",test_image_id[0][0],",",train_id," :",output)
                            output_dict[train_id] = [output.item(),train_attr]
                            #output_dict["attributes"] = tr_target
                            print(test_image_id[0][0],train_id)
                            #file.write("Cosine Similarity:",output)
                    # if train_i == 5000:
                    #     break
    # for k, v in output_dict.items():
    #     print(type(v))
    #     output_dict[k] = float(v)
    #     print(type(v))
    #import collections

    #sorted_dict = collections.OrderedDict(output_dict)
    #a = sorted(output_dict.items(), key=lambda x: x[1],reverse=True)
    a = sorted(output_dict.items(), key=lambda x: x[1][0],reverse=True)
    for idx, k in enumerate(a):
        if idx == n: break
        print(k)
    # f = open("a_file.txt", "w")
    # for item in a:
    #     f.write(item + "\n")

    #print(first_n_items)

            # cos = torch.nn.CosineSimilarity(dim=0)
            # output = cos(prediction, target)
            # print("Cosine Similarity:",output)
            # print("output size",output.size())
            # #dist = torch.pow(target - prediction, 2).sum(2)
            # #print("Dsitance Measure:",dist)
            # pdist = nn.PairwiseDistance(p=2)
            # pairwise_dist = pdist(target, prediction)
            # cdist = torch.cdist(target.float(), prediction.float(), p=2)
            # #print("pairwise_dist:",pairwise_dist)
            # #print("torch.cdist:",cdist)
            # print("Mean Cosine Similarity:",torch.mean(output))
            #print("Mean pairwise_dist:",torch.mean(pairwise_dist))
            #print("Mean torch.cdist:",torch.mean(cdist))

            # if i == 100:
            #     break
        #print(losses)

# model()
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'saturn'
    os.environ['MASTER_PORT'] = '29700'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    
    #torch.manual_seed(40)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    setup(int(sys.argv[1]), int(sys.argv[2]))
    #train(int(sys.argv[1]), int(sys.argv[2]))
    predict(int(sys.argv[1]), int(sys.argv[2]))

