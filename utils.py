import argparse
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchvision
import torchvision.models.detection.mask_rcnn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

# data path format
# - data_mfcc 
# -- 001
# -- 002
# -- ...

path = "./data_mfcc/"

class musicDatasetTrain(Dataset):
    
    def __init__(self, dataPath, data_num):
        super(musicDatasetTrain, self).__init__()
        np.random.seed(0)
        self.datas, self.num_classes = self.loadToMem(dataPath)
        self.data_num = data_num


    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
                    datas[idx] = []
                    for samplename in os.listdir(os.path.join(dataPath, alphaPath)):
                        filePath = os.path.join(dataPath, alphaPath,samplename)
                        #print(filePath)
                        datas[idx].append(np.load(filePath))
                    idx += 1
        print("finish loading training dataset to memory")
        return datas, idx
       

    def __getitem__(self, index):
        label = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        image1 = torch.Tensor(image1)
        image2 = torch.Tensor(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


    def __len__(self):
        return self.data_num
    

class musicDatasetTest(Dataset):

    def __init__(self, dataPath, times=200, way=20):
        np.random.seed(1)
        super(musicDatasetTest, self).__init__()
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)


    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
                    datas[idx] = []
                    for samplename in os.listdir(os.path.join(dataPath, alphaPath)):
                        filePath = os.path.join(dataPath, alphaPath, samplename)
                        #print(filePath)
                        datas[idx].append(np.load(filePath))
                    idx += 1
        print("finish loading training dataset to memory")
        return datas, idx


    def __len__(self):
        return self.times * self.way


    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])
        img1 = torch.Tensor(self.img1)
        img2 = torch.Tensor(img2)
        return img1, img2

if __name__ == "__main__":
     # test
     datapath = "data_mfcc"
     trainset = musicDatasetTrain(datapath,100)
     print(len(trainset))
     print(trainset[0])

    