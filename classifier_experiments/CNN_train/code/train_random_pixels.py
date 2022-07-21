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

from tqdm import tqdm

# set directory
os.chdir("/users/riya/race/classifier_experiments/CNN_train")

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
    
# preprocessing

def process_skeletonize(img, skeleton, 
                        image_size = (224, 224)):
    
    img = np.array(img)
    img = cv2.resize(img, image_size)
    
    # defining channel which will be duplicated late (in case it's not already with Image Folder??)
    channel = img[:,:,0]
    
    if skeleton is True:
        # thresholding (removing these pixels) can be done with this line
        # skeleton_channel[skeleton_channel < 20] = 0   
        channel[channel > 0] = 255       
        modified_img = skeletonize(channel, method='lee')
    
    elif skeleton is not True:
        modified_img = channel
    
    return img, modified_img

def shadow_threshold(c, operation, none_thresh, increment):
    if (c <= none_thresh): # just to write it explicitly
        c = c
    else:
        if (operation == 'add'):
            summed = c + increment
            if (summed > 255):
                summed = 255
            c = summed
        if (operation == 'subtract'):
            difference = c - increment
            if (difference < 0):
                difference = 0
            c = difference
    return c

def apply_threshold(image, operation, none_thresh, increment):
    
    img_arr = np.copy(image)
    
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            new_cell = shadow_threshold(img_arr[i][j], operation, none_thresh, increment) # final values chosen
            img_arr[i][j] = new_cell
    
    return img_arr

def randomly_distribute(img, skeleton, brighten_sum, # can try with multiple brighten_sums
                       none_thresh = 0, image_size = (224, 224)): # getting rid of the complete black
    
    # sticking with none_thresh = 0 for now, brightening everything except the background. 
    # could try some thresholding on which pixels we brighten, or thresholding by zeroing some pixels.
    
    # resize/skeletonize first, and then random distribution  
    img, modified_img = process_skeletonize(img, skeleton) # image size given
    
    # random distribution now 
    flattened_img = modified_img.flatten()
    random_img = np.random.permutation(flattened_img)
    modified_img2 = np.reshape(random_img, image_size)
        
    # now for brightening code, will be brightening ALL pixels above 0/20? (above 0, brightening everything)
    modified_img3 = apply_threshold(modified_img2, 'add', none_thresh, brighten_sum)
    
    img[:,:,0] = modified_img3
    img[:,:,1] = modified_img3
    img[:,:,2] = modified_img3
    
    img = Image.fromarray(img)

    return img


# train code, fix with new addition.

def train(data_dir, brighten_sum, experiment_name, skeleton=False,
          num_classes=2, batch_size=64, num_epochs=50, lr=0.001, image_size = (224, 224)): # 50 epochs for optimal performance
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # just this once using gpu1 on train0, bc 0 used.
    if device == 'cuda:1': # using all available gpus
        torch.cuda.empty_cache()
    if skeleton is True: 
        f_params = f'./outputs/checkpoints/{experiment_name}/model_random_pixels_brightened_by_{brighten_sum}_skeletonized_epoch{num_epochs}.pt'
        f_history = f'./outputs/histories/{experiment_name}/model_random_pixels_brightened_by_{brighten_sum}_skeletonized_epoch{num_epochs}.json'
        csv_name = f'./outputs/probabilities/{experiment_name}/random_pixels_brightened_by_{brighten_sum}_skeletonized_epoch{num_epochs}.csv'
    elif skeleton is False:
        f_params = f'./outputs/checkpoints/{experiment_name}/model_random_pixels_brightened_by_{brighten_sum}_epoch{num_epochs}.pt'
        f_history = f'./outputs/histories/{experiment_name}/model_random_pixels_brightened_by_{brighten_sum}_epoch{num_epochs}.json'
        csv_name = f'./outputs/probabilities/{experiment_name}/random_pixels_brightened_by_{brighten_sum}_epoch{num_epochs}.csv'
        
        
    # fix these transforms    
    train_transforms = transforms.Compose([transforms.Lambda(lambda img: randomly_distribute(img, skeleton, brighten_sum)), 
                                           # image size + none_thresh are pre-defined
                                           # transforms.Resize(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(25),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])]) # why this normalizing?
    
    test_transforms = transforms.Compose([transforms.Lambda(lambda img: randomly_distribute(img, skeleton, brighten_sum)),
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
    print(f'Skeletonize: {skeleton}')
    print(f'Brightened by: {brighten_sum} px')
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
    
    experiment_name = '#8(random_pixels)' # change depending on experiment
    data_dir = os.path.join('dataset')
    
    if not os.path.isdir(os.path.join('outputs', 'probabilities', experiment_name)):
        os.makedirs(os.path.join('outputs', 'probabilities', experiment_name))
    if not os.path.isdir(os.path.join('outputs', 'checkpoints', experiment_name)):
        os.makedirs(os.path.join('outputs', 'checkpoints', experiment_name))
    if not os.path.isdir(os.path.join('outputs', 'histories', experiment_name)):
        os.makedirs(os.path.join('outputs', 'histories', experiment_name))

    # experiment 8 part 1
        
    # original: most logical
    train(data_dir, 0, experiment_name, skeleton=False)
              
    # brightening increase
    train(data_dir, 20, experiment_name, skeleton=False)
    train(data_dir, 40, experiment_name, skeleton=False)
    train(data_dir, 60, experiment_name, skeleton=False)
    train(data_dir, 80, experiment_name, skeleton=False)
    train(data_dir, 100, experiment_name, skeleton=False)
    train(data_dir, 120, experiment_name, skeleton=False)
    
    # weird test, cause theoretically should have NO information now, since all intensities are constant + no structure
    train(data_dir, 0, experiment_name, skeleton=True)



