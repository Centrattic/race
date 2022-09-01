import os
import shutil

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from PIL import Image
import cv2

from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, Checkpoint, EpochScoring, EarlyStopping
from skorch.dataset import Dataset
from skorch.helper import predefined_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import warnings

from tqdm import tqdm

# set directory
os.chdir("/users/riya/race/classifier_experiments/CNN_train")

warnings.filterwarnings("ignore")

# model definitions

class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)
    
    
def skeletonize_images(img, skeleton, # the channel here was weird for half_skeletonize but didn't appear a training issue???
                image_size): # (224, 224) typically
    img = np.copy(img)
    img = np.array(img)
    
    img = cv2.resize(img, image_size)
    
    # defining channel which will be duplicated later (in case it's not already with Image Folder??)
    channel = img[:,:,0]
    
    if skeleton is True:
        # thresholding (removing these pixels) can be done with this line
        # skeleton_channel[skeleton_channel < 20] = 0   
        skeleton_channel = np.copy(channel)
        skeleton_channel[skeleton_channel > 0] = 255       
        modified_img = skeletonize(skeleton_channel, method='lee')
    
    elif skeleton is not True:
        modified_img = channel
    
    img_copy = np.copy(img) # original tri channeled
    
    img_copy[:,:,0] = modified_img
    img_copy[:,:,1] = modified_img
    img_copy[:,:,2] = modified_img
    
    img_copy = Image.fromarray(img_copy)
    
    return img_copy


def train(data_dir, experiment_name, skeleton,
          num_classes=2, batch_size=64, num_epochs=50, lr=0.001, image_size = (224, 224)): # 50 epochs for optimal performance
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 
    if device == 'cuda:1': # using all available gpus
        torch.cuda.empty_cache()

    f_params = f'./outputs/checkpoints/{experiment_name}/model_skeleton_{skeleton}.pt'
    f_history = f'./outputs/histories/{experiment_name}/model_skeleton_{skeleton}.json'
    csv_name = f'./outputs/probabilities/{experiment_name}/model_skeleton_{skeleton}.csv'
    
    train_transforms = transforms.Compose([transforms.Lambda
                                      (lambda img: skeletonize_images(img, skeleton, image_size)), 
                                           # transforms.Resize(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(25),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])]) # why this normalizing?
    
    test_transforms = transforms.Compose([transforms.Lambda
                                      (lambda img: skeletonize_images(img, skeleton, image_size)),
                                          # transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])

    train_folder = os.path.join(data_dir, 'train') # only training on segmentations      
    val_folder = os.path.join(data_dir, 'val')
    test_folder = os.path.join(data_dir, 'test')
    
    # I guess this automatically creates 3 channels
    train_dataset = datasets.ImageFolder(train_folder, train_transforms)
    val_dataset = datasets.ImageFolder(val_folder, test_transforms)
    test_dataset = datasets.ImageFolder(test_folder, test_transforms)
    
    print ("Train/Test/Val datasets have been created.")

    labels = np.array(train_dataset.samples)[:,1]
    
    # what even does the below code do?    
    labels = labels.astype(int) # idk what changed 6/17/22?? going from 0 and 1, to 1 and 2 labels. So I will subtract 1
    black_weight = 1 / len(labels[labels == 0]) 
    white_weight = 1 / len(labels[labels == 1])
    sample_weights = np.array([black_weight, white_weight])
    weights = sample_weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_dataset), replacement=True)

    print()
    print(f'Data Directory: {data_dir}')
    print(f'Experiment Name: {experiment_name}')
    print(f'Skeletonize: {skeleton}')
    print(f'Number of Classes: {num_classes}')
    print(f'Number of black eyes: {len(labels[labels == 0])}')
    print(f'Number of white eyes: {len(labels[labels == 1])}')
    print(f'Batch Size: {batch_size}')
    print(f'Number of Epochs: {num_epochs}')
    print(f'Initial Learning Rate: {lr}')
    print(f'Device: {device}')
    print()

    # maybe increase size of validation set??
    
    checkpoint = Checkpoint(monitor='valid_loss_best',
                            f_params=f_params,
                            f_history=f_history,
                            f_optimizer=None,
                            f_criterion=None)

    # accuracy on train/validation?
    
    train_acc = EpochScoring(scoring='accuracy',
                             on_train=True,
                             name='train_acc',
                             lower_is_better=False)

    early_stopping = EarlyStopping()

    callbacks = [checkpoint, train_acc, early_stopping]

    net = NeuralNetClassifier(PretrainedModel,
                              criterion=nn.CrossEntropyLoss,
                              lr=lr,
                              batch_size=batch_size,
                              max_epochs=num_epochs,
                              module__output_features=num_classes,
                              optimizer=optim.SGD,
                              optimizer__momentum=0.9,
                              iterator_train__num_workers=16,
                              iterator_train__sampler=sampler,
                              iterator_valid__shuffle=False,
                              iterator_valid__num_workers=16,
                              train_split=predefined_split(val_dataset),
                              callbacks=callbacks,
                              device=device)

    print ("Model is fitting. Thank you for your patience.")
    net.fit(train_dataset, y=None)

    print ("Model is performing inference. Results saved in probabilities folder.")

    img_locs = [loc for loc, _ in test_dataset.samples]
    test_probs = net.predict_proba(test_dataset)
    test_probs = [prob[0] for prob in test_probs] # probability of being black
    data = {'img_loc' : img_locs, 'probability' : test_probs}
    pd.DataFrame(data=data).to_csv(csv_name, index=False)
    
    print ("The code is done.")

    
# run train

if __name__ == '__main__':
    
    data_dir = os.path.join('dataset_given')
    experiment_name = 'given_dataset/base_models'
    
    if not os.path.isdir(os.path.join('outputs', 'probabilities', experiment_name)):
        os.makedirs(os.path.join('outputs', 'probabilities', experiment_name))
    if not os.path.isdir(os.path.join('outputs', 'checkpoints', experiment_name)):
        os.makedirs(os.path.join('outputs', 'checkpoints', experiment_name))
    if not os.path.isdir(os.path.join('outputs', 'histories', experiment_name)):
        os.makedirs(os.path.join('outputs', 'histories', experiment_name))

    # original model
   #  train(data_dir, experiment_name, skeleton=False)
    
    # skeletonized control
    train(data_dir, experiment_name, skeleton=True)
    
    # training 6 skeleton & shadow (no ring) models for experiment #2
    # train(data_dir, 45, [0,0] 'dark_center',skeleton=True, shadow = True, shadow_ring = False)