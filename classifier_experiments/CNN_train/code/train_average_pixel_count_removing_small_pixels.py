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
from utils_train import make_nonzero_dict, checksum, determine_image_race, average_pixel_count, nonzero_pixel_check_during_training

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

def train(data_dir, experiment_name, set_name,
          num_classes=2, batch_size=64, num_epochs=50, lr=0.001, image_size = (224, 224)): # 50 epochs for optimal performance
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") # just this once using gpu1 on train0, bc 0 used.
    if device == 'cuda:3': # using all available gpus
        torch.cuda.empty_cache()

    f_params = f'./outputs/checkpoints/{experiment_name}/{set_name}_model_normalized_nonzero_pixels_epoch{num_epochs}.pt'
    f_history = f'./outputs/histories/{experiment_name}/{set_name}_model_normalized_nonzero_pixels_epoch{num_epochs}.json'
    csv_name = f'./outputs/probabilities/{experiment_name}/{set_name}_normalized_nonzero_pixels_epoch{num_epochs}.csv'
    
    if set_name == "train_test_val_set":
        nonzero_dict_train = make_nonzero_dict('train')
        nonzero_dict_test = make_nonzero_dict('test')
        nonzero_dict_val = make_nonzero_dict('val')
    
        number_of_pixels_train = int(abs(nonzero_dict_train['skeletonized_white'] - 
                                         nonzero_dict_train['skeletonized_black'])) # skeleton images
        number_of_pixels_test = int(abs(nonzero_dict_test['skeletonized_white'] - 
                                        nonzero_dict_test['skeletonized_black'])) # skeleton images
        number_of_pixels_val = int(abs(nonzero_dict_val['skeletonized_white'] - 
                                        nonzero_dict_val['skeletonized_black'])) # skeleton images
    
    elif set_name == "full_set":
        nonzero_dict = make_nonzero_dict('full_set')
        number_of_pixels_train = int(abs(nonzero_dict['skeletonized_white'] - 
                                         nonzero_dict['skeletonized_black'])) # skeleton images
        number_of_pixels_test = number_of_pixels_train
        number_of_pixels_val = number_of_pixels_train
    # nonzero dict contains information for 1 channel, so no division.
    
    data_csv_path = "/users/riya/race/csv/image_race_data.csv"  
    race_data, checksum_dict = checksum(data_csv_path) # what if, since checksum is all 4546, it identifies ID as image not even in train set! Probably pretty rare.
    
    # fix these transforms    
    train_transforms = transforms.Compose([transforms.Lambda(lambda img: average_pixel_count(img, determine_image_race(img, race_data, checksum_dict), number_of_pixels_train)), 
                                           # image size + none_thresh are pre-defined
                                           # transforms.Resize(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(25),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])]) # why this normalizing?
    
    test_transforms = transforms.Compose([transforms.Lambda(lambda img: average_pixel_count(img, determine_image_race(img, race_data, checksum_dict), number_of_pixels_test)),
                                          # transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])
    
    val_transforms = transforms.Compose([transforms.Lambda(lambda img: average_pixel_count(img, determine_image_race(img, race_data, checksum_dict), number_of_pixels_val)),
                                          # transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])
    
    # testing code has an issue. Average pixel count will need to be computed for test images as well so that the model can be tested optimally (I should pass in that parameter). And what about val set, because model uses it to determine training time, although doesn't train from this ??

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
    
    # checking that counts are equal
    
    print ("Checking Nonzero Pixel Counts")
    
    train_tuple = nonzero_pixel_check_during_training(train_dataset)
    val_tuple = nonzero_pixel_check_during_training(val_dataset)
    test_tuple = nonzero_pixel_check_during_training(test_dataset)
    
    """black_non_zero = [0]
    white_non_zero = [0]
    
    tmp = iter(train_dataset)
    num_iterations = len(train_dataset.samples)

    for i in tqdm(range(num_iterations)):
        image, label = next(tmp)   
        run_img = np.array(image) # not sure where equal median number comes from? Ohh prob different/lower because normalizing.
        num_nonzero = int(np.count_nonzero(run_img)/3)

        if label == 0:
            black_non_zero = np.append(black_non_zero, num_nonzero)
        elif label == 1:
            white_non_zero = np.append(white_non_zero, num_nonzero)

    black_non_zero = black_non_zero[1:]
    white_non_zero = white_non_zero[1:]   """

    print()
    print(f'Data Directory: {data_dir}')
    print(f'Experiment Name: {experiment_name}')
    print(f'Train (Black, White) Median: {train_tuple}')
    print(f'Val (Black, White) Median: {val_tuple}')
    print(f'Test (Black, White) Median: {test_tuple}')
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
    train(data_dir, experiment_name, 'train_test_val_set') # exactly equal nonzero pixel count values
    train(data_dir, experiment_name, 'full_set') # non equal (exact) nonzero pixel count values
    
    # try normalizing val + train together?? Hmm idk. Full_set should be okay.
    
    # I guess there's not a lot of parameter's we're looking at right now. Would be totally different for random pixels.
