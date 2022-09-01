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

from utils_train import make_nonzero_dict, randomly_distribute

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

# train code, fix with new addition.

def train(data_dir, experiment_name, set_name, brighten_sum, average_nonzero_pixels = True, skeleton=False,
          num_classes=2, batch_size=64, num_epochs=50, lr=0.001, image_size = (224, 224)): # 50 epochs for optimal performance
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") # just this once using gpu1 on train0, bc 0 used.
    if device == 'cuda:2': # using all available gpus
        torch.cuda.empty_cache()
        
    f_params = f'./outputs/checkpoints/{experiment_name}/{set_name}_model_random_pixels_brightened_by_{brighten_sum}_epoch{num_epochs}.pt'
    f_history = f'./outputs/histories/{experiment_name}/{set_name}_model_random_pixels_brightened_by_{brighten_sum}_epoch{num_epochs}.json'
    csv_name = f'./outputs/probabilities/{experiment_name}/{set_name}_random_pixels_brightened_by_{brighten_sum}_epoch{num_epochs}.csv'
        
    if set_name == "train_test_val_set": # no longer supporting - not going to use!
        nonzero_dict_train = make_nonzero_dict('train')
        nonzero_dict_test = make_nonzero_dict('test')
        nonzero_dict_val = make_nonzero_dict('val')
    
        number_of_pixels_train = int((nonzero_dict_train['non-skeletonized_white'] + nonzero_dict_train['non-skeletonized_black'])/2)
        number_of_pixels_test = int((nonzero_dict_test['non-skeletonized_white'] + nonzero_dict_test['non-skeletonized_black'])/2)
        number_of_pixels_val = int((nonzero_dict_val['non-skeletonized_white'] + nonzero_dict_val['non-skeletonized_black'])/2)
    
    elif set_name == "full_set":
        nonzero_dict = make_nonzero_dict('full_set')
        number_of_pixels_train = int((nonzero_dict['non-skeletonized_white'] + nonzero_dict['non-skeletonized_black'])/2)
        number_of_pixels_test = number_of_pixels_train
        number_of_pixels_val = number_of_pixels_train        

    train_transforms = transforms.Compose([transforms.Lambda(lambda img: randomly_distribute(img, skeleton, average_nonzero_pixels, number_of_pixels_train, brighten_sum)), 
                                           # image size + none_thresh are pre-defined
                                           # transforms.Resize(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(25),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])]) # why this normalizing?
    
    test_transforms = transforms.Compose([transforms.Lambda(lambda img: randomly_distribute(img, skeleton, average_nonzero_pixels, number_of_pixels_test, brighten_sum)),
                                          # transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])
    
    val_transforms = transforms.Compose([transforms.Lambda(lambda img: randomly_distribute(img, skeleton, average_nonzero_pixels, number_of_pixels_val, brighten_sum)),
                                          # transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])

    train_folder = os.path.join(data_dir, 'train') # only training on segmentations      
    val_folder = os.path.join(data_dir, 'val')
    test_folder = os.path.join(data_dir, 'test')
    
    # I guess this automatically creates 3 channels
    train_dataset = datasets.ImageFolder(train_folder, train_transforms)
    val_dataset = datasets.ImageFolder(val_folder, val_transforms)
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
    
    print ("Checking Nonzero Pixel Counts")
    
#     train_tuple = nonzero_pixel_check_during_training(train_dataset)
#     val_tuple = nonzero_pixel_check_during_training(val_dataset)
#     test_tuple = nonzero_pixel_check_during_training(test_dataset)

    print()
    print(f'Data Directory: {data_dir}')
    print(f'Skeletonize: {skeleton}')
    print(f'Set Name: {set_name}')
    print(f'Nonzero Pixel Counts (Train, Test, Val): {number_of_pixels_train, number_of_pixels_test, number_of_pixels_val}')
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
    
    experiment_name = 'given_dataset/#2(random_pixels_constant_pixel_count)' # change depending on experiment
    data_dir = os.path.join('dataset_given')
    
    if not os.path.isdir(os.path.join('outputs', 'probabilities', experiment_name)):
        os.makedirs(os.path.join('outputs', 'probabilities', experiment_name))
    if not os.path.isdir(os.path.join('outputs', 'checkpoints', experiment_name)):
        os.makedirs(os.path.join('outputs', 'checkpoints', experiment_name))
    if not os.path.isdir(os.path.join('outputs', 'histories', experiment_name)):
        os.makedirs(os.path.join('outputs', 'histories', experiment_name))

    # set: train test val set
    
    # differing train test val image # of pixels (illogical for sure now!!!!)
    # train(data_dir, experiment_name, 'train_test_val_set',
    
    # set: full set
    
    # brightening increase
    # alreay done 
    # train(data_dir, experiment_name, 'full_set', 0)
    # train(data_dir, experiment_name, 'full_set', 20)
    # train(data_dir, experiment_name, 'full_set', 40)
    # train(data_dir, experiment_name, 'full_set', 60)
    # train(data_dir, experiment_name, 'full_set', 80)
    # train(data_dir, experiment_name, 'full_set', 100)
    train(data_dir, experiment_name, 'full_set', 120)