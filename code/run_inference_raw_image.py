"""
Author: Praveer SIngh
Date: 04/04/2021

Model: pre-trained ResNet18
Dataset: Training on Train/Val/Test splits from BPD+Full_ROP dataset from Pete
oversampling of BPD class
originally trained using MONAI version: 0.3.0rc4
"""
import os
import pandas as pd
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision.models

from torchvision.transforms import Lambda, Normalize 

from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import compute_roc_auc
from monai.transforms import (
    AddChannel,
    AsChannelFirst,
    Compose,
    LoadPNG,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity, Transpose, 
    LoadImage,
    ToTensor,
    Resize,
)
from monai.utils import set_determinism
from dataset import ROPDatasetRGBImage
from utils import BPD_CNN, train, val, test

print("Environment onfiguration is:")
print_config()
new_datadir = "./../Posterior"

dict_label2idx = {0.0:int(0),1.0:int(1)}

df_train = pd.read_csv('train_plus_cleaned.csv', usecols=['imagename','o236'])
#df_train = df_train.replace({'o236': dict_label2idx})
df_train['o236'] = df_train['o236'].astype(int)
df_train = df_train.rename(columns={"imagename": "imageName", "o236": "label"})

class_count_train = np.unique(df_train['label'], return_counts=True)[1]

df_val = pd.read_csv('val_plus_cleaned.csv', usecols=['imagename','o236'])
#df_val = df_val.replace({'o236': dict_label2idx})
df_val['o236'] = df_val['o236'].astype(int)
df_val = df_val.rename(columns={"imagename": "imageName", "o236": "label"})

class_count_val = np.unique(df_val['label'], return_counts=True)[1]

df_test_plus_unclean = pd.read_csv('test.csv', usecols=['imagename','o236'])
#df_test = df_test.replace({'o236': dict_label2idx})
df_test_plus_unclean['o236'] = df_test_plus_unclean['o236'].astype(int)
df_test_plus_unclean = df_test_plus_unclean.rename(columns={"imagename": "imageName", "o236": "label"})

class_count_test_plus_unclean = np.unique(df_test_plus_unclean['label'], return_counts=True)[1]

df_test = pd.read_csv('test_plus_cleaned.csv', usecols=['imagename','o236'])
#df_test = df_test.replace({'o236': dict_label2idx})
df_test['o236'] = df_test['o236'].astype(int)
df_test = df_test.rename(columns={"imagename": "imageName", "o236": "label"})

class_count_test = np.unique(df_test['label'], return_counts=True)[1]

train_paths_old = df_train['imageName'].tolist()
train_image_files_list = [os.path.join(new_datadir, i) for i in train_paths_old] # combine the image name with the new datapath 

val_paths_old = df_val['imageName'].tolist() 
val_image_files_list = [os.path.join(new_datadir, i) for i in val_paths_old] # combine the image name with the new datapath 

test_plus_unclean_paths_old = df_test_plus_unclean['imageName'].tolist() 
test_plus_unclean_image_files_list = [os.path.join(new_datadir, i) for i in test_plus_unclean_paths_old] # combine the image name with the new datapath 

test_paths_old = df_test['imageName'].tolist() 
test_image_files_list = [os.path.join(new_datadir, i) for i in test_paths_old] # combine the image name with the new datapath 

list_all_images = []

train_image_class = df_train['label'].tolist()
train_image_class_list = []

val_image_class = df_val['label'].tolist()
val_image_class_list = []

test_plus_unclean_image_class = df_test_plus_unclean['label'].tolist()
test_plus_unclean_image_class_list = []

test_image_class = df_test['label'].tolist()
test_image_class_list = []

for elem in os.listdir('./../Posterior/'):
    list_all_images.append(new_datadir + '/' + elem)

train_image_files_list_updated = []
val_image_files_list_updated = []
test_plus_unclean_image_files_list_updated = []
test_image_files_list_updated = []

for i, elem in enumerate(train_image_files_list):
    if elem in list_all_images:
        train_image_files_list_updated.append(elem)
        train_image_class_list.append(train_image_class[i])

for i, elem in enumerate(val_image_files_list):
    if elem in list_all_images:
        val_image_files_list_updated.append(elem)
        val_image_class_list.append(val_image_class[i])

for i, elem in enumerate(test_plus_unclean_image_files_list):
    if elem in list_all_images:
        test_plus_unclean_image_files_list_updated.append(elem)
        test_plus_unclean_image_class_list.append(test_plus_unclean_image_class[i])

for i, elem in enumerate(test_image_files_list):
    if elem in list_all_images:
        test_image_files_list_updated.append(elem)
        test_image_class_list.append(test_image_class[i])


"""
transforms 
"""
train_transforms = Compose(
    [
        ScaleIntensity(minv=0.0,maxv=1.0),
        RandRotate(range_x=15, prob=0.1, keep_size=True), # low probability for rotation 
        RandFlip(spatial_axis=0, prob=0.5),# left right flip 
        RandFlip(spatial_axis=1, prob=0.5), # horizontal flip
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        AsChannelFirst(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

val_transforms = Compose(
    [
        ScaleIntensity(minv=0.0,maxv=1.0),
        AsChannelFirst(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

test_transforms = Compose(
    [
        ScaleIntensity(minv=0.0,maxv=1.0),
        AsChannelFirst(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
# Dataloader
#BPD Class % prevelance in trainset
# 0: 57.36%, 1: 42.64%
weights_class = [1.74,2.35] # manually define the weights for each class 
weights_images = [weights_class[train_image_class_item] for train_image_class_item in train_image_class_list]

# Weighted sampler for unbalanced dataset
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_images, len(train_image_class_list), replacement=True)   
batch_size = 64
batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last = True)

train_ds = ROPDatasetRGBImage(train_image_files_list_updated, train_image_class_list, train_transforms)
train_loader = torch.utils.data.DataLoader(train_ds, batch_sampler = batch_sampler, num_workers=4)

val_ds = ROPDatasetRGBImage(val_image_files_list_updated, val_image_class_list, val_transforms)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=4)

test_plus_unclean_ds = ROPDatasetRGBImage(test_plus_unclean_image_files_list_updated, test_plus_unclean_image_class_list, test_transforms)
test_plus_unclean_loader = torch.utils.data.DataLoader(test_plus_unclean_ds, batch_size=batch_size, num_workers=4)

test_ds = ROPDatasetRGBImage(test_image_files_list_updated, test_image_class_list, test_transforms)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=4)

# Model
model = BPD_CNN().cuda()
device = torch.device("cuda:0")
model.to(device)


loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
val_interval = 1 # evaluate accuracy after each epoch

model_dir = "./models/"
epoch_num = 8

# Training
best_metric_val = 0.0
best_metric_test = 0.0
best_metric_epoch = 0
epoch_loss_values = list()
metric_values = list()
'''
for epoch in range(epoch_num):
    print("-" * 50)
    print(f"epoch {epoch + 1}/{epoch_num}")
    train(epoch, model, loss_function, optimizer, train_loader, device, model_dir)
    print("....................Validation....................")
    best_metric_val, best_metric_epoch = val(epoch, model, val_loader, device, best_metric_val, best_metric_epoch, model_dir)

print(f"Best_metric_Validation: {best_metric_val:.4f} at epoch: {best_metric_epoch}")
'''
print("....................Running best model on Test Set..................")
model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth")))
best_metric_test = test(model, test_loader, device, model_dir)
best_metric_test_plus_unclean = test(model, test_plus_unclean_loader, device, model_dir)
print(f"Testing completed, best_auc_test_plus_unclean_performnce: {best_metric_test_plus_unclean:.4f} and best_auc_test_performance: {best_metric_test:.4f}")
