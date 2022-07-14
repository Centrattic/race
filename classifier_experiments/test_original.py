import os
import shutil
import hickle
import re

import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import numpy as np
import PIL
from PIL import Image, ImageChops, ImageDraw

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

from tqdm import tqdm
from numba import jit, cuda
from utils import systemic_brightening
from train_random_pixels import PretrainedModel
import warnings


# set directory
os.chdir("/users/riya/race/classifier_experiments")

# ignore warnings
warnings.filterwarnings("ignore")

# import model
segmentation_classifier = keras.models.load_model('models/MIMIC-256x25680-20-split-resnet-Float16_2-race_detection_rop_seg_data_rop_seg-0.001_20220321-054140_epoch:011.hdf5')


# useful functions ---------------------------------------

def image_from_id(img_path, path_name):
    arr = np.array(Image.open(img_path + path_name))
    resized = cv2.resize(arr, (256,256))
    channels = np.repeat(resized[:, :, np.newaxis], 3, axis=2).reshape((256,256,3))
    
    return channels


def get_race_from_id(img_id, race_csv_path):

    race_data = pd.read_csv(race_csv_path)
    img_row = race_data.loc[race_data['image_id'] == int(img_id)] # they both must be ints
    img_row = img_row.reset_index(drop=True) # for .at to work
    img_race = img_row.at[0,'race']
    
    return img_race


# --------------------------------------------------------

def test(data_dir, thresh_type, intensity_change, brighten_sum, experiment_name, model_path,
         skeleton=False, none_thresh = 20, num_classes = 2, image_size = (224, 224)):
    
    os.chdir("/users/riya/race/classifier_experiments/CNN_train/")
    race_data_path = "/users/riya/race/csv/image_race_data.csv"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device == 'cuda:1':
        torch.cuda.empty_cache()
    
    csv_name = f'../predictions/experiment1_plus_systemic_brightening/{experiment_name}/' + str(intensity_change) + '_by_' + str(brighten_sum) + '.csv'
    
    print(csv_name)
    
    test_transforms = transforms.Compose([transforms.Lambda(lambda img: systemic_brightening
                                                            (img, skeleton, thresh_type, intensity_change, brighten_sum)),
                                          # transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])
    
    test_folder = os.path.join(data_dir, 'test')
    test_dataset = datasets.ImageFolder(test_folder, test_transforms)    
    
    # Pytorch load model
    model = PretrainedModel(num_classes)
    model.load_state_dict(torch.load(model_path))
    
    # load into Skorch
    net = NeuralNetClassifier(model, 
                              criterion=nn.CrossEntropyLoss,
                              device=device)
    
    net.initialize() # bc I am not using net.fit (training the model)
    
    print ("Model Loaded + Initialized", model_path)
    
    img_locs = [loc for loc, _ in test_dataset.samples]
    img_ids = [re.findall(r'\d+', loc)[1] for loc in img_locs] # instantaneous basically
    
    #print(img_ids)
    
    img_race = [get_race_from_id(img_id, race_data_path) for img_id in tqdm(img_ids)]
    
    # print(img_race)
    print("Number of Images: " + str(len(img_race)))
    
    print ("Starting Predictions")
    
    test_probs = net.predict_proba(test_dataset)
    
    print ("Predictions Done")
    test_probs = [prob[1] for prob in test_probs] # probability of being white
    data = {'img_id' : img_ids, 'race': img_race, 'probability_' + str(brighten_sum) : test_probs}
    pd.DataFrame(data=data).to_csv(csv_name, index=False)
    
    print ("Code Done")