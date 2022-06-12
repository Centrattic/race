import os

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

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from tqdm import tqdm

os.chdir("/users/riya/race/classifier_experiments/CNN_train")


# model definition

class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)

    
# preprocessing

def shadow_regions(img, skeleton, shadow, radius, region, image_size = (224, 224)):
    
    img = np.array(img)
    img = cv2.resize(img, image_size)
    
    # defining channel which will be duplicated late (in case it's not already with Image Folder??)
    channel = img[:,:,0]

    if skeleton is True:
        # can binarize all 3 channels, but will go 1 at a time
        channel[channel > 0] = 255       
        modified_img = skeletonize(channel, method='lee')
    
    if shadow is True:
        # developing mask that darkens center portion
        center_mask = np.full(image_size, 255, dtype=np.uint8) 
        # radius i changes, center, color, fill is the same
        cv2.circle(center_mask, (int(image_size[0]/2), int(image_size[0]/2)), radius, (0, 0, 0), -1)

        # developing mask that darkens background region
        back_mask = cv2.bitwise_not(center_mask)

        if (region == 'dark_center'):
            modified_img = cv2.bitwise_or(channel, channel, mask=center_mask)

        if (region == 'dark_background'):
            modified_img = cv2.bitwise_or(channel, channel, mask=back_mask)
    
    if skeleton is not True and shadow is not True: # if condition here for clarity     
        modified_img = channel
        
    img[:,:,0] = modified_img
    img[:,:,1] = modified_img
    img[:,:,2] = modified_img
    
    img = Image.fromarray(img)

    return img


# train code

def train(data_dir, radius, region, skeleton=False, shadow = False, num_classes=2, batch_size=1, num_epochs=10, lr=0.001):
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    if device == 'cuda:1': # using all available gpus
        torch.cuda.empty_cache()
    if skeleton is True: # Experiment #2: skeleton true, shadow true
        shadow = True
        f_params = f'./outputs/checkpoints/model_shadow_regions_{region}_{radius}_skeletonized.pt'
        f_history = f'./outputs/histories/model_shadow_regions_{region}_{radius}_skeletonized.json'
        csv_name = f'./outputs/probabilities/shadow_regions_{region}_{radius}_skeletonized.csv'
    elif shadow is True: # Experiment #3: skeleton false, shadow true
        f_params = f'./outputs/checkpoints/model_shadow_regions_{region}_{radius}.pt'
        f_history = f'./outputs/histories/model_shadow_regions_{region}_{radius}.json'
        csv_name = f'./outputs/probabilities/shadow_regions_{region}_{radius}.csv'
    else: # Original training: skeleton false, shadow false
        f_params = f'./outputs/checkpoints/model_original.pt'
        f_history = f'./outputs/histories/model_original.json'
        csv_name = f'./outputs/probabilities/original.csv'
        
    train_transforms = transforms.Compose([transforms.Lambda(lambda img: shadow_regions(img, skeleton,
                                                                                shadow, radius,
                                                                                region)), # image size pre-defined
                                           # transforms.Resize(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(25),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])]) # why this normalizing?

    test_transforms = transforms.Compose([transforms.Lambda(lambda img: shadow_regions(img, skeleton,
                                                                                shadow, radius,
                                                                                region)),
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

    
    labels = np.array(train_dataset.samples)[:,1]
    
    # what even does the below code do?    
    labels = labels.astype(int)
    black_weight = 1 / len(labels[labels == 0])
    white_weight = 1 / len(labels[labels == 1])
    sample_weights = np.array([black_weight, white_weight])
    weights = sample_weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_dataset), replacement=True)

    print()
    print(f'Data Directory: {data_dir}')
    print(f'Skeletonize: {skeleton}')
    print(f'Shadow: {shadow}')
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

    net.fit(train_dataset, y=None)

    img_locs = [loc for loc, _ in test_dataset.samples]
    test_probs = net.predict_proba(test_dataset)
    test_probs = [prob[0] for prob in test_probs]
    data = {'img_loc' : img_locs, 'probability' : test_probs}
    pd.DataFrame(data=data).to_csv(csv_name, index=False)

    
# running the train (no memory error pls)

if __name__ == '__main__':
    if not os.path.isdir(os.path.join('outputs', 'probabilities')):
        os.makedirs(os.path.join('outputs', 'probabilities'))
    if not os.path.isdir(os.path.join('outputs', 'checkpoints')):
        os.makedirs(os.path.join('outputs', 'checkpoints'))
    if not os.path.isdir(os.path.join('outputs', 'histories')):
        os.makedirs(os.path.join('outputs', 'histories'))

    data_dir = os.path.join('dataset')

    train(data_dir, 0, 0)
    
    # training 4 skeleton + shadow models for experiment #2
    train(data_dir, 45, 'dark_center',skeleton=True, shadow = True)
    train(data_dir, 45, 'dark_background',skeleton=True, shadow = True)
    train(data_dir, 90, 'dark_center',skeleton=True, shadow = True)
    train(data_dir, 90, 'dark_background',skeleton=True, shadow = True)