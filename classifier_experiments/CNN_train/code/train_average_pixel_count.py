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
    

# train code, simple bc functions do the hard lifting.

def train(data_dir, experiment_name,
          num_classes=2, batch_size=64, num_epochs=50, lr=0.001, image_size = (224, 224)): # 50 epochs for optimal performance
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # just this once using gpu1 on train0, bc 0 used.
    if device == 'cuda:1': # using all available gpus
        torch.cuda.empty_cache()

    f_params = f'./outputs/checkpoints/{experiment_name}/model_normalized_nonzero_pixels_epoch{num_epochs}.pt'
    f_history = f'./outputs/histories/{experiment_name}/model_normalized_nonzero_pixels_epoch{num_epochs}.json'
    csv_name = f'./outputs/probabilities/{experiment_name}/normalized_nonzero_pixels_epoch{num_epochs}.csv'
    
    nonzero_dict = make_nonzero_dict()
    number_of_pixels = number_of_pixels = nonzero_dict['skeletonized_white'] - nonzero_dict['skeletonized_black'] # skeleton images
    number_of_pixels = int(abs(number_of_pixels))
    # nonzero dict contains information for 1 channel, so no division.
    
    data_csv_path = "/users/riya/race/csv/image_race_data.csv"  
    race_data, checksum_dict = checksum(data_csv_path)
    
    # fix these transforms    
    train_transforms = transforms.Compose([transforms.Lambda(lambda img: average_pixel_count(img, determine_image_race(img, race_data, checksum_dict), number_of_pixels)), 
                                           # image size + none_thresh are pre-defined
                                           # transforms.Resize(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(25),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])]) # why this normalizing?
    
    test_transforms = transforms.Compose([transforms.Lambda(lambda img: average_pixel_count(img, determine_image_race(img, race_data, checksum_dict), nonzero_dict)),
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
    
    # checking that counts are equal
    
    print ("Determining if Nonzero Pixel Counts are Equal")
    
    black_non_zero = [0]
    white_non_zero = [0]

    for i, _ in train_dataset.samples:
        run_img = np.array(Image.open(i))
        num_nonzero = np.count_nonzero(run_img)
        if 'black' in i:
            black_non_zero = np.append(black_non_zero, num_nonzero)
        elif 'white' in i:
            white_non_zero = np.append(white_non_zero, num_nonzero)

    black_non_zero = black_non_zero[1:]
    white_non_zero = white_non_zero[1:]
    

    print()
    print(f'Data Directory: {data_dir}')
    print(f'Experiment Name: {experiment_name}')
    print(f'White Median: {np.median(white_non_zero)}')
    print(f'Black Median: {np.median(black_non_zero)}')
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
    
    experiment_name = '#11(average_pixel_count)' # change depending on experiment
    data_dir = os.path.join('dataset_full')
    
    if not os.path.isdir(os.path.join('outputs', 'probabilities', experiment_name)):
        os.makedirs(os.path.join('outputs', 'probabilities', experiment_name))
    if not os.path.isdir(os.path.join('outputs', 'checkpoints', experiment_name)):
        os.makedirs(os.path.join('outputs', 'checkpoints', experiment_name))
    if not os.path.isdir(os.path.join('outputs', 'histories', experiment_name)):
        os.makedirs(os.path.join('outputs', 'histories', experiment_name))

    # experiment 11 part 1
        
    # original: most logical
    



