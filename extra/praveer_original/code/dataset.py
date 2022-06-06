import os
import pandas as pd
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision.models
import cv2
from torchvision.transforms import Lambda, Normalize 


def Preprocess(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB) #converting to LAB color-space
    l, a, b = cv2.split(lab) #splitting LAB to different channels
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #applying CLAHE to L-channel
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b)) #merge CLAHE enhance L-channel with a and b channel
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) #switching back to RGB color space
    return img

class RaceDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index], self.image_files[index]

class RaceDatasetRawImage(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms
        self.inputs_dtype = torch.float32

    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, index):
        x_img = cv2.imread(self.image_files[index])
        x_img = Preprocess(x_img)
        x_img = cv2.resize(x_img, (480, 640))
        
        return self.transforms(x_img).type(self.inputs_dtype), self.labels[index], self.image_files[index]
