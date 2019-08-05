import argparse, torch
import numpy as np
from torch.utils.data import Dataset
from torch import nn
from torch.autograd import Variable
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from PIL import ImageFile,Image

class Dataset_filename(Dataset):
    """
    return image,filename
    """
    def __init__(self, root_dir,w=None,h=None,mean=0.5,stdv=1,need_filename=False):
        self.root_dir = root_dir
        self.filenames=[]
        self.labels=[]
        self.imagespath=[]
        self.mean=mean
        self.stdv=stdv
        self.w,self.h=w,h
        self.need_filename=need_filename

        for filepath in os.listdir(root_dir):
            for image in os.listdir(os.path.join(root_dir,filepath)):
                self.filenames.append(image)
                self.labels.append(int(filepath))
                self.imagespath.append(os.path.join(root_dir,filepath,image))
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        X = np.zeros((3,self.w,self.h),dtype='float32')
        try:
            #print(input_dir+'/'+filepath)
            image=Image.open(self.imagespath[idx])
            #print('read over')
            image=np.array(image.resize((self.w,self.h)),dtype='float32')
            image=(image / 255.0) 
            image=(image-self.mean)/self.stdv
            X[0, :, :] = image[:,:,0]
            X[1, :, :] = image[:,:,1]
            X[2, :, :] = image[:,:,2]
        except:
            pass
        if self.need_filename:
            return X,self.labels[idx],self.filenames[idx]
        return X,self.labels[idx]