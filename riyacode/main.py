"""
Author: Praveer SIngh
Date: 08/22/2021

Model: pre-trained ResNet18
Dataset: Training on Train/Val/Test splits from ROP dataset from Pete
oversampling of Black class
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
from dataset import RaceDataset
from utils import Race_CNN, train, val, test # utils.py, import is defining these functions (basically running utils.py)

#Race class% prevelance in trainset
# Black(0): 35.96%, White(1): 64.04%
weights_class = [2.78,1.56] # manually define the weights for each class

# where do these weights come from, why do it manually? Basically making each = 100 for normalization (weight-balance in sklearn)

set_determinism(seed=0)
new_datadir = "./../dataset/segmentations" # changed from ./../segmented, appears they are now in this new directory
csv_dir = "./../csv"
model_dir = "./../models/"
batch_size = 64
learning_rate = 1e-4
cuda_device = "cuda:0"
epoch_num = 30
criterion_function = torch.nn.BCEWithLogitsLoss()
val_interval = 1 # evaluate accuracy after each epoch

print("Environment configuration is:")
print_config()

dict_label2idx = {'black':int(0),'white':int(1)}

# could check if exists in segmentations here versus later

df_train = pd.read_csv(csv_dir + '/train.csv', usecols=['image_id','race'])
df_train = df_train.replace({'race': dict_label2idx})
df_train = df_train.rename(columns={"image_id": "imageName", "race": "label"})

class_count_train = np.unique(df_train['label'], return_counts=True)[1]

df_val = pd.read_csv(csv_dir + '/val.csv', usecols=['image_id','race'])
df_val = df_val.replace({'race': dict_label2idx})
df_val = df_val.rename(columns={"image_id": "imageName", "race": "label"})

class_count_val = np.unique(df_val['label'], return_counts=True)[1]

df_test = pd.read_csv(csv_dir + '/test.csv', usecols=['image_id','race'])
df_test = df_test.replace({'race': dict_label2idx})
df_test = df_test.rename(columns={"image_id": "imageName", "race": "label"})

class_count_test = np.unique(df_test['label'], return_counts=True)[1]


# why called old?
train_paths_old = [str(f) for f in df_train['imageName']] #split extension as segmented images are all .png
train_image_files_list = [os.path.join(new_datadir, i+".bmp") for i in train_paths_old] # combine the image name with the new datapath 

val_paths_old = [str(f) for f in df_val['imageName']] #split extension as segmented images are all .png
val_image_files_list = [os.path.join(new_datadir, i+".bmp") for i in val_paths_old] # combine the image name with the new datapath 

test_paths_old = [str(f) for f in df_test['imageName']] #split extension as segmented images are all .png
test_image_files_list = [os.path.join(new_datadir, i+".bmp") for i in test_paths_old] # combine the image name with the new datapath 

# handling if all images not segmented, filtering only segmented ones

list_all_images = []

train_image_class = df_train['label'].tolist()
train_image_class_list = []

val_image_class = df_val['label'].tolist()
val_image_class_list = []

test_image_class = df_test['label'].tolist()
test_image_class_list = []

for elem in os.listdir(new_datadir):
    list_all_images.append(new_datadir + '/' + elem) # getting file path to every single file in segmentations

train_image_files_list_updated = []
val_image_files_list_updated = []
test_image_files_list_updated = []

for i, elem in enumerate(train_image_files_list): # i = index, elem = element
    if elem in list_all_images:
        train_image_files_list_updated.append(elem) # what's actually in directory (segmented)
        train_image_class_list.append(train_image_class[i]) # labels for what's actually in directory (segmented)

for i, elem in enumerate(val_image_files_list):
    if elem in list_all_images:
        val_image_files_list_updated.append(elem)
        val_image_class_list.append(val_image_class[i])

for i, elem in enumerate(test_image_files_list):
    if elem in list_all_images:
        test_image_files_list_updated.append(elem)
        test_image_class_list.append(test_image_class[i])


"""
transforms 
"""

# preprocessing images before DataLoader

# why are different commands used on train (rotating, and flipping the data) vs on validation and test.

# retina images - ukbb, uk bio bank over 5-10 yrs
# * resize images?

train_transforms = Compose(
    [
        LoadPNG(image_only=True),
        AddChannel(), # why add channel? these have 3?
        ScaleIntensity(), 
        RandRotate(range_x=15, prob=0.1, keep_size=True), # low probability for rotation 
        RandFlip(spatial_axis=0, prob=0.5),# left right flip 
        RandFlip(spatial_axis=1, prob=0.5), # horizontal flip
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5), 
        ToTensor(),
        Lambda(lambda x: torch.cat([x, x, x], 0)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std of imagenet dataset, try with and without
    ]
)

val_transforms = Compose(
    [
        LoadPNG(image_only=True),
        # Resize((480,640)),
        AddChannel(), 
        ScaleIntensity(),
        ToTensor(),
        Lambda(lambda x: torch.cat([x, x, x], 0)),
        # ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]
)

# no need to transform data in test

test_transforms = Compose(
    [
        LoadPNG(image_only=True),
        # Resize((480,640)),
        AddChannel(), 
        ScaleIntensity(),
        ToTensor(),
        Lambda(lambda x: torch.cat([x, x, x], 0)),
        # ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]
)
# Dataloader

# all the weights that should be applied to the images in order, 2.78, 1.56, 1.56, 2.78, etc. etc. 
# why not on test and validation, validation specifically?
weights_images = [weights_class[train_image_class_item] for train_image_class_item in train_image_class_list] 

# Weighted sampler for unbalanced dataset
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_images, len(train_image_class_list), replacement=True)   
batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last = True)

train_ds = RaceDataset(train_image_files_list_updated, train_image_class_list, train_transforms) # train transforms is sequential of methods
train_loader = torch.utils.data.DataLoader(train_ds, batch_sampler = batch_sampler, num_workers=4) # what is batch sampler?

val_ds = RaceDataset(val_image_files_list_updated, val_image_class_list, val_transforms)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=4)

test_ds = RaceDataset(test_image_files_list_updated, test_image_class_list, test_transforms)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=4)

# Model
model = Race_CNN().cuda()
device = torch.device(cuda_device)
model.to(device)

loss_function = criterion_function
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# Training
best_metric_val = 0.0
best_metric_test = 0.0
best_metric_epoch = 0
epoch_loss_values = list()
metric_values = list()

for epoch in range(epoch_num):
    print("-" * 50)
    print(f"epoch {epoch + 1}/{epoch_num}")
    train(epoch, model, loss_function, optimizer, train_loader, device, model_dir)
    print("....................Validation....................")
    best_metric_val, best_metric_epoch = val(epoch, model, val_loader, device, best_metric_val, best_metric_epoch, model_dir)

print(f"Best_metric_Validation: {best_metric_val:.4f} at epoch: {best_metric_epoch}")
print("....................Running best model on Test Set..................")
model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth")))
best_metric_test = test(model, test_loader, device, model_dir)
print(f"Testing completed, best_auc_test_performnce: {best_metric_test:.4f}")

# what variables, parameters are defined here instead of in utils
# (it appears all tunable hyperparameters defined here, while model-related optimizer, loss, etc. in utils.py)
