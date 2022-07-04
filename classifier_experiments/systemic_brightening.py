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
import torch
from torch import nn
import torchvision.models
import cv2

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import matplotlib.patches as patches

from tqdm import tqdm
from numba import jit, cuda
from utils import systemic_brightening

# set directory
os.chdir("/users/riya/race/classifier_experiments") # which one? yep

# import model
segmentation_classifier = keras.models.load_model('models/MIMIC-256x25680-20-split-resnet-Float16_2-race_detection_rop_seg_data_rop_seg-0.001_20220321-054140_epoch:011.hdf5')


# run through classifier

jit(target ="cuda")                         
def predict_on_images(img_path, preds_df, colname,
                     skeleton, thresh_type, intensity_change, brighten_sum,
                     csv_name = "brightened_predictions"):

    id_arr = []
    img_arr = []
    
    img_files = os.listdir(dataset_path)
    
    for i in tqdm(range(len(img_files))):
        arr = np.array(Image.open(img_path + img_files[i]))
        resized = cv2.resize(arr, (256,256)) # must use 256
        channels = np.repeat(resized[:, :, np.newaxis], 3, axis=2).reshape((256,256,3))

        modified_img = systemic_brightening(channels, skeleton, thresh_type, intensity_change, brighten_sum,
                                           image_size = (256, 256))
        modified_img = np.array(modified_img).reshape((1,256,256,3))
        img_arr = np.append(img_arr, modified_img)
                       
        # getting id     
        img_id = re.findall(r'\d+', img_files[i])
        id_arr = np.append(id_arr, img_id)
    
    preds_df['id'] = id_arr
  
    # getting prediction    
    num_images = len(id_arr)  
    preds_arr = [0] * num_images
    prediction = segmentation_classifier(img_arr)
    
    for i in range(num_images):
        preds_arr[i] = prediction.numpy()[i, 1] # returning the white prediction for each image
  
    preds_df[colname] = preds_arr           
    preds_df.to_csv(preds_path + csv_name + ".csv")

    
# run images
                 
all_predictions = pd.DataFrame(columns = ['id', '30', '60', '90', '120', '150']) # from id I can get race

dataset_path = "/users/riya/race/dataset/segmentations/"
preds_path = "/users/riya/race/classifier_experiments/predictions/experiment1_plus_systemic_brightening/"
                  
predict_on_images(dataset_path, all_predictions, '30', False, 'below', 'brighten', 30)
predict_on_images(dataset_path, all_predictions, '60', False, 'below', 'brighten', 60)
predict_on_images(dataset_path, all_predictions, '90', False, 'below', 'brighten', 90)
predict_on_images(dataset_path, all_predictions, '120', False, 'below', 'brighten', 120)         
predict_on_images(dataset_path, all_predictions, '150', False, 'below', 'brighten', 150)     

predict_on_images(dataset_path, all_predictions, '30', False, 'below', 'dull', 30, csv_name = "dulled_predictions")
predict_on_images(dataset_path, all_predictions, '60', False, 'below', 'dull', 60, csv_name = "dulled_predictions")
predict_on_images(dataset_path, all_predictions, '90', False, 'below', 'dull', 90, csv_name = "dulled_predictions")
predict_on_images(dataset_path, all_predictions, '120', False, 'below', 'dull', 120, csv_name = "dulled_predictions")         
predict_on_images(dataset_path, all_predictions, '150', False, 'below', 'dull', 150, csv_name = "dulled_predictions")                          