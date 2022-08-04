# imports
import os
import shutil

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from PIL import Image
import cv2
import re
import glob
import random

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
from sklearn import metrics
from tqdm import tqdm

# beginner functions ---------------------------------------------------------------------------------------------------------

def get_colvalue_from_id(img_id, image_data, colname):
    img_row = image_data.loc[image_data['image_id'] == int(img_id)] # they both must be ints
    img_row = img_row.reset_index(drop=True) # for .at to work
    img_value = img_row.at[0,colname]
    
    return img_value   

def channeled_image_from_id(img_path, path_name, image_size): # will reshape to 3 channels! careful!
    arr = np.array(Image.open(img_path + path_name))
    # print(arr.shape)
    resized = cv2.resize(arr, (image_size[1], image_size[0])) # cv2.resize takes (width, height), which is (640, 480)!!
    # print(resized.shape)
    channels = np.repeat(resized[:, :, np.newaxis], 3, axis=2).reshape((image_size[0], image_size[1],3))
    # print(channels.shape)

    return channels

def count_nonzero(img, threshold):
    count_img = np.array(img)
    nonzero_count = np.count_nonzero(count_img > threshold)
    
    return nonzero_count

def process_skeletonize(img, skeleton, # the channel here was weird for half_skeletonize but didn't appear a training issue???
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
    
    return img, channel, modified_img


def perform_shadow(c, operation, increment):

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


def shadow_threshold(c, operation, thresh_type, none_thresh, increment):
    
    if (thresh_type == 'below'):
        if (c <= none_thresh): # just to write it explicitly
            c = c
        else:
            c = perform_shadow(c, operation, increment)
    elif (thresh_type == 'above'):
        if (c > none_thresh): # > or >=? prob >
            c = c
        elif (c > 0): # just so background doesn't get impacted
            c = perform_shadow(c, operation, increment)
            
    return c

def apply_threshold(image, operation, none_thresh, increment, thresh_type = 'below'): # defaulting
    
    img_arr = np.copy(image)
    
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            new_cell = shadow_threshold(img_arr[i][j], operation, thresh_type, none_thresh, increment) # final values chosen
            img_arr[i][j] = new_cell
    
    return img_arr

def substitute_channels(img, substitute):
    
    img_copy = np.copy(img)
    
    img_copy[:,:,0] = substitute
    img_copy[:,:,1] = substitute
    img_copy[:,:,2] = substitute
    
    img_copy = Image.fromarray(img_copy)
    
    return img_copy


def image_center_from_id(QA_csv, img_id, img_size):
    img_row = QA_csv[QA_csv['image_id'] == img_id] # check if string
    
    y_og = img_row['y'].reset_index(drop=True)[0]
    x_og = img_row['x'].reset_index(drop=True)[0]
    
    x_pos = (80 + y_og) * img_size[1]/640 # x size is 640, cropped that way
    y_pos = (x_og)* img_size[0]/480 # y size is 480 (also, height is first in img size tuple, so the 0)
    
    disk_center = (int(x_pos), int(y_pos))
    
    return disk_center

def load_QA_csv(optic_disk_csv = "/users/riya/race/optic_disk/DeepROP/quality_assurance/QA.csv"):
    QA_csv = pd.read_csv(optic_disk_csv)
    
    QA_csv.columns.values[0] = "image_id"
    QA_csv.columns = QA_csv.columns.to_series().apply(lambda x: x.strip())
    QA_csv[['image_id', 'Full path', 'x', 'y', 'is_posterior']]

    QA_csv = QA_csv[QA_csv['is_posterior'] == True]
    
    return QA_csv

def load_eye_key_csv(eye_key_csv = "/users/riya/race/csv/image_eye_key.csv",
                    race_data_csv = "/users/riya/race/csv/image_race_data.csv"):
    
    eye_info_csv = pd.read_csv(eye_key_csv)
    race_data_csv = pd.read_csv(race_data_csv)
    
    eye_info_csv = eye_info_csv[eye_info_csv.image_id.isin(race_data_csv['image_id'])]
    
    return eye_info_csv

def make_nonzero_dict(set_name, preds_path = "/users/riya/race/classifier_experiments/nonzero_count/"):
    
    # all these refer to nonzero values for radiuses that include the whole image, not sections of the image
    
    if set_name == 'train':
        nonskel_name = 'train_set_nonskeletonized_nonzero_count_above_0.csv'
        skel_name = 'train_set_skeletonized_nonzero_count_above_0.csv'
    
    if set_name == 'test':
        nonskel_name = 'test_set_nonskeletonized_nonzero_count_above_0.csv'
        skel_name = 'test_set_skeletonized_nonzero_count_above_0.csv'
    
    if set_name == 'val':
        nonskel_name = 'val_set_nonskeletonized_nonzero_count_above_0.csv'
        skel_name = 'val_set_skeletonized_nonzero_count_above_0.csv'
    
    if set_name == 'optic_disk_set':
        nonskel_name = 'final_nonskeletonized_nonzero_count_above_0.csv' # this file does not exist
        skel_name = 'final_skeletonized_nonzero_count_above_0.csv'
    
    if set_name == 'full_set':
        nonskel_name = 'full_set_nonskeletonized_nonzero_count_above_0.csv'
        skel_name = 'full_set_skeletonized_nonzero_count_above_0.csv'
        
    nonskel_0 = pd.read_csv(preds_path + nonskel_name) # bad. probably wrong
    skel_0 = pd.read_csv(preds_path + skel_name) 
    # new version appears correct (with all 3181 train set images))
    
    nonskel_0_white = nonskel_0[nonskel_0['race'] == 'white']
    nonskel_0_black = nonskel_0[nonskel_0['race'] == 'black']

    skel_0_white = skel_0[skel_0['race'] == 'white']
    skel_0_black = skel_0[skel_0['race'] == 'black']
    
    # will use median for now. can change to mean if significant.
    
    nonzero_dict = {'skeletonized_white': np.median(skel_0_white['0-318']),
                'skeletonized_black': np.median(skel_0_black['0-318']),
                'non-skeletonized_white': np.median(nonskel_0_white['0-318']),
                'non-skeletonized_black': np.median(nonskel_0_black['0-318'])}
    
    return nonzero_dict

def nonzero_pixel_check_during_training(data_loader):
    black_non_zero = [0]
    white_non_zero = [0]
    
    tmp = iter(data_loader)
    num_iterations = len(data_loader.samples)

    for i in tqdm(range(num_iterations)):
        image, label = next(tmp)   
        run_img = np.array(image)[:,:,0] # not sure where equal median number comes from? Ohh prob different/lower because normalizing.
        num_nonzero = int(np.count_nonzero(run_img))

        if label == 0:
            black_non_zero = np.append(black_non_zero, num_nonzero)
        elif label == 1:
            white_non_zero = np.append(white_non_zero, num_nonzero)

    black_non_zero = black_non_zero[1:]
    white_non_zero = white_non_zero[1:]
    
    return (np.median(black_non_zero), np.median(white_non_zero))

def make_area_sorted_dict(numLabels, stats):

    area_info = [0] * numLabels
    area_sorted_dict = {}
    
    for i in range(0, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        area_info[i] = area
    
    area_dict = dict(pd.value_counts(area_info)) 
    for key in sorted(area_dict.keys()):
        area_sorted_dict[key] = area_dict[key]
    
    return area_sorted_dict

# medium functions --------------------------------------------------------------------------------------------------------------------

def checksum(data_csv_path, data_path = "/users/riya/race/dataset/segmentations/"): # images with equal checksums are VERY close duplicates
    
    # done with images in original shape of (480, 640, 1)
    
    data_csv = pd.read_csv(data_csv_path)
    
    checksum_arr = []
    id_arr = []

    # optic_csv_path = "../../optic_disk/DeepROP/quality_assurance/QA.csv"
    # returns only images with optics disks

    for i in tqdm(data_csv['image_id']):
        img_compare = np.array(Image.open(data_path + str(i) + '.bmp'))
        
        # print (img_compare.shape)
        all_sum = np.concatenate(img_compare).sum()
        col_sum = img_compare[:,100:240].sum()
        
        checksum_arr.append(all_sum + col_sum)
        id_arr.append(i) 
    
    checksum_dict = {id_arr[i]: checksum_arr[i] for i in range(len(id_arr))}
    
    return data_csv, checksum_dict

def determine_image_side_view(img, eye_info_csv, checksum_dict): 
    
    # return whether eye is left or right & eye view (posterior, nasal, etc.)
    
    id_og = '' # original id of image, for comparison
    
    img_og = np.array(img) # our original image
    img_og = img_og[:,:,0]
    
    checksum_og = np.concatenate(img_og).sum() + img_og[:,100:240].sum()
    
    id_og = list(checksum_dict.keys())[list(checksum_dict.values()).index(checksum_og)] # finding id for which checksum_og matches
    
    eye_direction = get_colvalue_from_id(id_og, eye_info_csv, 'eye')
    eye_view = get_colvalue_from_id(id_og, eye_info_csv, 'view')
    
    eye_tuple = (eye_direction, eye_view)
    
    return eye_tuple

def determine_image_center(img, img_size, QA_csv, checksum_dict): # using optic disk
    
    id_og = '' # original id of image, for comparison
    
    img_og = np.array(img) # our original imageimg_og
    img_og = img_og[:,:,0]
    
    checksum_og = np.concatenate(img_og).sum() + img_og[:,100:240].sum()
    
    id_og = list(checksum_dict.keys())[list(checksum_dict.values()).index(checksum_og)] # finding id for which checksum_og matches
    
    # all images are of size 480 x 480
    
    disk_center = image_center_from_id(QA_csv, id_og, img_size)
    
    """img_row = QA_csv[QA_csv['img_id'] == id_og] # check if string
    
    y_og = img_row['y'].reset_index(drop=True)[0]
    x_og = img_row['x'].reset_index(drop=True)[0]
    
    x_pos = (80 + y_og) * img_size[1]/640 # x size is 640, cropped that way
    y_pos = (x_og)* img_size[0]/480 # y size is 480 (also, height is first in img size tuple, so the 0)
    
    # never subtract 480 because from the top :( and image orientation is naturally from the top)
    
    disk_center = (int(x_pos), int(y_pos)) # for (224, 224) image that will be created soon"""
    
    return disk_center

def determine_image_race(img, race_data, checksum_dict): # using optic disk
    
    id_og = '' # original id of image, for comparison
    
    img_og = np.array(img) 
    img_og = img_og[:,:,0] # our original image, of size (480, 640, 1)
    
    # print (img_og.shape)
    
    checksum_og = np.concatenate(img_og).sum() + img_og[:,100:240].sum()
    
    id_og = list(checksum_dict.keys())[list(checksum_dict.values()).index(checksum_og)] # finding id for which checksum_og matches
        
    img_race = get_colvalue_from_id(id_og, race_data, 'race')
   
    return img_race


# These region masks create center masks (center hidden), which can then be inverted to make back masks easily.

def ring_region_mask(disk_center, ring_radiuses, # region will be radius (0, ..region)
                     image_size = (224, 224)):

    # white background
    center_mask = np.full(image_size, 255, dtype=np.uint8)
    # large black circle on outside
    cv2.circle(center_mask, disk_center, ring_radiuses[1], (0, 0, 0), -1)
    # smaller white circle on inside
    
    if ring_radiuses[0] > 0: # center hidden aah!
        cv2.circle(center_mask, disk_center, ring_radiuses[0], (255, 255, 255), -1)
    
    return center_mask


def multiple_ring_mask(disk_center, num_rings, ring_radiuses,
                       image_size = (224, 224)):
    
    # ex. ring_radiuses: [[0, 15], [75, 90]]
    
    center_mask = np.full(image_size, 0, dtype=np.uint8) 

    for i in range(num_rings):
        ring_mask = np.full(image_size, 0, dtype=np.uint8)  
        # for each of the radiuses given dark around the outer circle
        cv2.circle(ring_mask, disk_center, ring_radiuses[i][1], (255, 255, 255), -1) 
        cv2.circle(ring_mask, disk_center, ring_radiuses[i][0], (0,0, 0), -1) # the white for that region
    
        center_mask = center_mask + ring_mask 

    # idk why exactly this is needed...? 
    # probably related to the fact that adding the original center_mask does NOT make sense, but appears to work
    center_mask = cv2.bitwise_not(center_mask) 
    
    # back_mask = cv2.bitwise_not(center_mask)
    # return cv2.bitwise_or(img, img, mask=back_mask)
    
    return center_mask

def isolating_macula_mask(eye_tuple,
                       image_size = (224, 224)):
    
    # mask will select the macular regions of the images
    
    eye_direction = eye_tuple[0]
    eye_view = eye_tuple[1]
    
    # white background
    select_macula = np.full(image_size, 255, dtype=np.uint8)
    
    half_x = int(image_size[0]/2)
    half_y = int(image_size[1]/2)
    
    if eye_direction == 'os': # left
        if eye_view == 'posterior' or eye_view == 'nasal' or eye_view == 'temporal':  
            cv2.rectangle(select_macula, (0,image_size[1]), (half_x, 0), (0, 0, 0), -1) 
    elif eye_direction == 'od': # right 
        if eye_view == 'posterior' or eye_view == 'nasal' or eye_view == 'temporal':
            cv2.rectangle(select_macula, (half_x,image_size[1]), (image_size[0], 0), (0, 0, 0), -1) 
            
    if eye_view == 'inferior':
        cv2.rectangle(select_macula, (0,image_size[1]), (image_size[0], half_y), (0, 0, 0), -1)     
    elif eye_view == 'superior':
        cv2.rectangle(select_macula, (0,half_y), (image_size[0], 0), (0, 0, 0), -1) 
    
    return select_macula

def average_pixel_count_random_images(img_channel, median_number_of_pixels, # all images 224 x 224
                    image_size = (224, 224)):
    
    # averaging count by removing low intensity pixels from white images
    # number_of_pixels = nonzero_dict['nonskeletonized_white'] - nonzero_dict['nonskeletonized_black'] # skeleton images
    # number_of_pixels = int(abs(number_of_pixels))
    # nonzero dict contains information for 1 channel, so no division.
    
    num_nonzero_img = np.count_nonzero(img_channel) # num pixels in 1 channel
    channel_flatten = np.array(img_channel.flatten())
    
    img_df = pd.DataFrame(index=range(len(channel_flatten)),columns=range(2))
    img_df['img_pixels'] = channel_flatten
    img_df.reset_index(inplace=True) # index colname      
                          
    pixels_to_modify = num_nonzero_img - median_number_of_pixels
    
     # add pixels, num_nonzero_img smaller
       
    img_df_zero = img_df[img_df['img_pixels'] == 0]
    img_df_nonzero = img_df[img_df['img_pixels'] != 0] # we want to stratify for PIVs to add
    test_size = abs(pixels_to_modify)/len(img_df_nonzero) 
    
    print(len(img_df_nonzero))

    _, X_test, _, y_test = train_test_split(img_df_nonzero['img_pixels'], img_df_nonzero['index'], test_size = test_size, stratify = img_df_nonzero['img_pixels'], random_state = 42)
       
        # X_test has img_pixel values that I need to add
      
    if pixels_to_modify <= 0: # if equal to 0 nothing will happen
        for i in range(pixels_to_modify): # sanity check       
            img_df_zero[i] = X_test[i] # then we'll append + sort 
    
    elif pixels_to_modify > 0:
        
        print(len(y_test), pixels_to_modify)
        assert len(y_test) == pixels_to_modify
        for i in range(len(y_test)): # 
            img_df_nonzero.loc[img_df_nonzero['index'] == y_test_i, 'img_pixels'] = 0   
                        
        concat_df = pd.concat([img_df_nonzero, img_df_zero]) # should be no duplicate index values
        concat_df = concat_df.sort_values(by='index', ascending=True)
        
        assert len(concat_df) == len(img_df)
        assert np.unique(concat_df['index']) == len(concat_df)
        
    img_flatten2 = concat_df['img_pixels']
    reshapen_img = img_flatten2.reshape(image_size)
    
    return reshapen_img    

def determine_image_info(number_of_pixels, area_sorted_dict):
    remaining_pixels = number_of_pixels
    
    for i in area_sorted_dict.keys(): # keys in the right sort order
        included_pixels = area_sorted_dict[i] * i 
        remaining_pixels -= included_pixels
            
        if remaining_pixels < 0: # we've reached our i limit
            limit_reached = i
            modulus_pixels = remaining_pixels + included_pixels
            number_vessels = int(np.floor(modulus_pixels/limit_reached))
            vessel_remainder = (modulus_pixels % limit_reached) # remainder, shld be int
            return limit_reached, number_vessels, vessel_remainder # return vs. the break (for loop indent)
        
def make_removed_components_mask(limit_reached, number_vessels, vessel_remainder, number_of_pixels, 
                                 numLabels, labels, stats): # exists per image 
    counter = 0
    mask = np.zeros((224, 224)).astype("uint8")
    randint_arr = [0.5] * numLabels # cause there'll never be a decimal label
    test_arr = [0]
    
    for i in range(0, numLabels): # not accounting for vessel fraction just yet
        
        img_index = 0.5
        while img_index in randint_arr: # will always be at first
            img_index = random.randint(0, numLabels-1) # retry man
        
        randint_arr[i] = img_index
 
        # above is inefficient code!
        
        area = stats[img_index, cv2.CC_STAT_AREA]
        
        if area < limit_reached:
            componentMask = (labels == img_index).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)

        elif area == limit_reached:
            if counter < number_vessels:
                test_arr = np.append(test_arr, img_index)
                componentMask = (labels == img_index).astype("uint8") * 255
                mask = cv2.bitwise_or(mask, componentMask)
                counter+=1
            # only happen when all limit reached vessels are gone. 
            # Then adding to counter, so vessel remainder happens only once
            elif counter == number_vessels:
                componentMask = (labels == img_index).astype("uint8") * 255
                num_pixels_remove = np.count_nonzero(componentMask) - vessel_remainder    
                a,b = np.nonzero(componentMask) # getting arrays of (a,b) nonzero indices
                index = np.random.choice(len(a), num_pixels_remove, replace=False)
                componentMask[a[index], b[index]] = 0
                mask = cv2.bitwise_or(mask, componentMask)
                
                counter+=1; # out of this elif
                
    final_mask = mask

    return final_mask     

# lambda applied functions --------------------------------------------------------------------------------------------------------------
def average_nonzero_pixels_with_vessels(img, img_race, number_of_pixels, connectivity,
                                       image_size = (224, 224)):
    img_resize, img_channel, img_skel_channel = process_skeletonize(img, True, image_size) # (224, 224)   
    
    if img_race == 'white':
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_skel_channel, 
                                                                               cv2.CV_32S, connectivity=connectivity)

        area_sorted_dict = make_area_sorted_dict(numLabels, stats)
        limit_reached, number_vessels, vessel_remainder = determine_image_info(number_of_pixels, area_sorted_dict)
        final_mask = make_removed_components_mask(limit_reached, number_vessels, vessel_remainder, 
                                                  number_of_pixels, numLabels, labels, stats)
        modified_channel = img_skel_channel - final_mask

    elif img_race == 'black':
        modified_channel = img_skel_channel # no change
        
    # print(np.count_nonzero(img_skel_channel), np.count_nonzero(modified_channel))
    final_img = substitute_channels(img_resize, modified_channel)

    # print(np.count_nonzero(img_channel), np.count_nonzero(img_skel_channel) * 3, np.count_nonzero(final_img))
    
    return final_img

def average_pixel_count(img, img_race, number_of_pixels, # all images 224 x 224
                    image_size = (224, 224)):
    
    # averaging count by removing low intensity pixels from white images
    # number_of_pixels = nonzero_dict['skeletonized_white'] - nonzero_dict['skeletonized_black'] # skeleton images
    # number_of_pixels = int(abs(number_of_pixels))
    # nonzero dict contains information for 1 channel, so no division.

    img, channel, modified_channel = process_skeletonize(img, True, image_size) # yes skeletonize ALWAYS

    if img_race == 'white':
        pixel_dict = dict(enumerate(channel.flatten(), 1)) # chooses one channel, they will be duplicated
        pixel_dict = {key - 1:val for key, val in pixel_dict.items() if val != 0}

        sorted_dict = {}
        sorted_keys = sorted(pixel_dict, key = pixel_dict.get)
        for w in sorted_keys:
            sorted_dict[w] = pixel_dict[w]

        # skeletonize img
        img_flatten = channel.flatten()
        skeleton_img_flatten = modified_channel.flatten()

        i = 0

        for key, value in sorted_dict.items(): # should go in sorted order, from min to max PIV
            assert img_flatten[key] == value # sanity check
            if i < number_of_pixels: # not <= 
                if skeleton_img_flatten[key] != 0: # pixel hasn't been removed in skeletonization
                    skeleton_img_flatten[key] = 0 # for white images, removing pixels
                    i+=1

        skeleton_channel = skeleton_img_flatten.reshape(image_size)
    
    elif img_race == 'black':
        skeleton_channel = modified_channel

    final_img = substitute_channels(img, skeleton_channel)
    
    return final_img    
    

def macula_focus(img, skeleton, eye_tuple, brighten_sum, region, # eye_tuple has (eye_direction, eye_view)
                     none_thresh = 0, image_size = (224, 224)):
    
    img, _ , modified_img = process_skeletonize(img, skeleton, image_size) # image size given
    
    # develop mask
    
    select_macula = isolating_macula_mask(eye_tuple)
    mask_macula = cv2.bitwise_not(select_macula)
    
    if (region == 'show_macula'): # could be for ring or not for ring
        modified_img2 = cv2.bitwise_or(modified_img, modified_img, mask=select_macula)
        
    elif (region == 'hide_macula'):
        # masking the center region
        modified_img2 = cv2.bitwise_or(modified_img, modified_img, mask=mask_macula)
    
    # now for brightening code
    modified_img3 = apply_threshold(modified_img2, 'add', none_thresh, brighten_sum, thresh_type = 'below') # yay default. Not dulling.
    final_img = substitute_channels(img, modified_img3)
    
    return final_img

def systemic_brightening(img, skeleton, thresh_type, intensity_change, brighten_sum,
                         none_thresh = 20, image_size = (224, 224)):
    # see if thicker vessels (no skeleton) influence black prediction (should be yes)
    # skeletonizing will show if it is about thickness or RELATIVE thickness, probably relative?
    # for skeletonizing, I will subtract from every vessel since brightest is 255. I'll code that later, it's a secondary part
    
    # I could try Thickening major vessels (> 20 pixels) 
    # I could also try thickening just minor vessels (<20 pixels) and seeing how that works out,
    # depending on race
    
    img, _ , modified_img = process_skeletonize(img, skeleton, image_size) # do I wanna threshold with skeletonization?
    
    # brightening code, brightening all pixels ABOVE 20
    if (thresh_type == 'below'): # ignoring pixels below none_thresh
        if (intensity_change == 'brighten'):
            modified_img2 = apply_threshold(modified_img, 'add', none_thresh, brighten_sum, thresh_type = 'below')
        if (intensity_change == 'dull'): # dull it
            modified_img2 = apply_threshold(modified_img, 'subtract', none_thresh, brighten_sum, thresh_type = 'below')
    
    # brightening code, brightening all pixels BELOW 20 (smaller vessels), HMM work on this experiment???
    if (thresh_type == 'above'): # ignoring pixels above none_thresh, not impacting them
        if (intensity_change == 'brighten'):
            modified_img2 = apply_threshold(modified_img, 'add', none_thresh, brighten_sum, thresh_type = 'above')
        if (intensity_change == 'dull'): # dull it
            modified_img2 = apply_threshold(modified_img, 'subtract', none_thresh, brighten_sum, thresh_type = 'above')
    
    final_img = substitute_channels(img, modified_img2)
    
    return final_img
                         

def randomly_distribute(img, skeleton, average_nonzero_pixels, median_number_of_pixels, brighten_sum, # can try with multiple brighten_sums
                       none_thresh = 0, image_size = (224, 224)): # getting rid of the complete black
    
    # for number of pixels we use non_skeletonized full_set + train/test/val sets.
    
    # sticking with none_thresh = 0 for now, brightening everything except the background. 
    # could try some thresholding on which pixels we brighten, or thresholding by zeroing some pixels.
    
    # resize/skeletonize first, and then random distribution  
    img, _ , modified_img = process_skeletonize(img, skeleton, image_size) # image size given
    
    # random distribution now 
    flattened_img = modified_img.flatten()
    random_img = np.random.permutation(flattened_img)
    modified_img2 = np.reshape(random_img, image_size)
    
    # averaging nonzero pixels
    if average_nonzero_pixels:
        modified_img3 = average_pixel_count_random_images(modified_img2, median_number_of_pixels)
    elif not average_nonzero_pixels:
        modified_img3 = modified_img2
        
    # now for brightening code, will be brightening ALL pixels above 0/20? (above 0, brightening everything)
    
    modified_img4 = apply_threshold(modified_img3, 'add', none_thresh, brighten_sum, thresh_type = 'below') # yay default
      
    final_img = substitute_channels(img, modified_img3)

    return final_img

def half_skeletonize(img, disk_center, skeleton_radiuses, region,
                    image_size = (224, 224)): 
    
    
    img, channel, skeleton_channel = process_skeletonize(img, True, image_size) # definitely skeletonizing!
    
    # create masks to shadow channel & skeleton_channel
    center_mask = np.full(image_size, 255, dtype=np.uint8)
    # large black circle on outside
    cv2.circle(center_mask, disk_center, skeleton_radiuses[1], (0, 0, 0), -1)
    # smaller white circle on inside
    cv2.circle(center_mask, disk_center, skeleton_radiuses[0], (255, 255, 255), -1)
    
    back_mask = cv2.bitwise_not(center_mask)
    
    if (region == 'skeleton_center'): # could be for ring or not for ring
        channel = cv2.bitwise_or(channel, channel, mask=center_mask)
        skeleton_channel = cv2.bitwise_or(skeleton_channel, skeleton_channel, mask=back_mask)

    if (region == 'skeleton_background'):
        # masking the center region
        channel = cv2.bitwise_or(channel, channel, mask=back_mask)
        skeleton_channel = cv2.bitwise_or(skeleton_channel, skeleton_channel, mask=center_mask)
    
    final_channel = channel + skeleton_channel
    
    final_img = substitute_channels(img, final_channel)

    return final_img

def shadow_regions(img, skeleton, disk_center, shadow, radius, 
                   shadow_ring, num_rings, ring_radiuses, region, 
                   image_size = (224, 224)):
    
    img = np.array(img)
    img = cv2.resize(img, image_size)
    
    # defining channel which will be duplicated late (in case it's not already with Image Folder??)
    channel = img[:,:,0]
    
    if skeleton is True:
        # can binarize all 3 channels, but will go 1 at a time
        channel[channel > 0] = 255       
        modified_img = skeletonize(channel, method='lee')
        
    elif skeleton is not True:
        modified_img = channel
    
    if shadow is True: # want to do either shadow or shadow_ring, not both
        
        if shadow_ring is True:
            # ring_radiuses is [inner_radius, outer_radius]
            
            # for cases of multiple rings, we're just going to pop over to another function lol. 
            # ring_radiuses will be array of arrays.
            if num_rings > 1: 
                center_mask = multiple_ring_mask(disk_center, num_rings, ring_radiuses,
                                                image_size = (224, 224))
        
            elif num_rings <= 1: # 1 ring only or no ring, only 1 ring really applies here
                # developing mask that darkens ring portion
                center_mask = np.full(image_size, 255, dtype=np.uint8) 
                # radius i changes, center, color, fill is the same
                cv2.circle(center_mask, disk_center, ring_radiuses[1], (0, 0, 0), -1)
                # adding circle to darken inside region
                cv2.circle(center_mask, disk_center, ring_radiuses[0], (255,255, 255), -1)
        
        elif shadow_ring is not True:
            # developing mask that darkens center portion
            center_mask = np.full(image_size, 255, dtype=np.uint8)
            # radius i changes, center, color, fill is the same
            cv2.circle(center_mask, disk_center, radius, (0, 0, 0), -1) # disk_center received from optic disk segmenter, tuple

        # developing mask that darkens background region (same in case of ring)
        back_mask = cv2.bitwise_not(center_mask)

        if (region == 'dark_center'): # could be for ring or not for ring
            modified_img2 = cv2.bitwise_or(modified_img, modified_img, mask=center_mask)
            
        if (region == 'dark_background'):
            modified_img2 = cv2.bitwise_or(modified_img, modified_img, mask=back_mask)
            
    elif shadow is not True: # if condition here for clarity     
        modified_img2 = modified_img
        
    img[:,:,0] = modified_img2
    img[:,:,1] = modified_img2
    img[:,:,2] = modified_img2
    
    img = Image.fromarray(img)

    return img