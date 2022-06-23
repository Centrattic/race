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
def checksum(optic_csv_path):
    
    # Turns out that around 31 of the images are actual duplicates! I looked at 10 and saw that they were. 
    # Checksum is getting the best of it, then.
    
    checksum_arr = []
    id_arr = []

    # optic_csv_path = "../../optic_disk/DeepROP/quality_assurance/QA.csv"
    data_compare_path = "/users/riya/race/dataset/segmentations/"

    QA_csv = pd.read_csv(optic_csv_path)

    QA_csv.columns.values[0] = "img_id"
    QA_csv.columns = QA_csv.columns.to_series().apply(lambda x: x.strip())
    QA_csv[['img_id', 'Full path', 'x', 'y', 'is_posterior']]

    QA_csv = QA_csv[QA_csv['is_posterior'] == True]

    for i in tqdm(QA_csv['img_id']):
        img_compare = np.array(Image.open(data_compare_path + str(i) + '.bmp'))
        all_sum = np.concatenate(img_compare).sum()
        col_sum = img_compare[:,100:240].sum()
        
        checksum_arr.append(all_sum + col_sum)
        id_arr.append(i)
    
    checksum_dict = {id_arr[i]: checksum_arr[i] for i in range(len(id_arr))}
    
    return QA_csv, checksum_dict

def determine_image_center(img, img_size, QA_csv, checksum_dict): # using optic disk
    
    id_og = '' # original id of image, for comparison
    
    img_og = np.array(img) # our original image
    img_og = img_og[:,:,0]
    
    checksum_og = np.concatenate(img_og).sum() + img_og[:,100:240].sum()
    
    id_og = list(checksum_dict.keys())[list(checksum_dict.values()).index(checksum_og)] # finding id for which checksum_og matches
    
    # all images are of size 480 x 480
    
    img_row = QA_csv[QA_csv['img_id'] == id_og] # check if string
    
    y_og = img_row['y'].reset_index(drop=True)[0]
    x_og = img_row['x'].reset_index(drop=True)[0]
    
    x_pos = (80 + y_og) * img_size[1]/640 # x size is 640, cropped that way
    y_pos = (x_og)* img_size[0]/480 # y size is 480 (also, height is first in img size tuple, so the 0)
    
    disk_center = (int(x_pos), int(y_pos)) # for (224, 224) image that will be created soon
    
    return disk_center

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

# train code, fix with new addition.

def train(data_dir, radius, num_rings, ring_radiuses, region, skeleton=False, shadow = False, shadow_ring = False,
          num_classes=2, batch_size=64, num_epochs=50, lr=0.001, image_size = (224, 224)): # 50 epochs for optimal performance
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # just this once using gpu1 on train0, bc 0 used.
    if device == 'cuda:0': # using all available gpus
        torch.cuda.empty_cache()
    if skeleton is True: # Experiment #2: skeleton true, shadow true
        if shadow is True:
            if shadow_ring is True:  
                if num_rings > 1: # accounting only for num_rings = 2 (or radiuses only up to 2 are included)
                    f_params = f'./outputs/checkpoints/model_shadow_rings_{region}_{ring_radiuses[0][0]}_{ring_radiuses[0][1]}_{ring_radiuses[1][0]}_{ring_radiuses[1][1]}_skeletonized_epoch{num_epochs}.pt'
                    f_history = f'./outputs/histories/model_shadow_rings_{region}_{ring_radiuses[0][0]}_{ring_radiuses[0][1]}_{ring_radiuses[1][0]}_{ring_radiuses[1][1]}_skeletonized_epoch{num_epochs}.json'
                    csv_name = f'./outputs/probabilities/shadow_rings_{region}_{ring_radiuses[0][0]}_{ring_radiuses[0][1]}_{ring_radiuses[1][0]}_{ring_radiuses[1][1]}_skeletonized_epoch{num_epochs}.csv'
                elif num_rings <= 1:
                    f_params = f'./outputs/checkpoints/model_shadow_rings_{region}_{ring_radiuses[0]}_{ring_radiuses[1]}_skeletonized_epoch{num_epochs}.pt'
                    f_history = f'./outputs/histories/model_shadow_rings_{region}_{ring_radiuses[0]}_{ring_radiuses[1]}_skeletonized_epoch{num_epochs}.json'
                    csv_name = f'./outputs/probabilities/shadow_rings_{region}_{ring_radiuses[0]}_{ring_radiuses[1]}_skeletonized_epoch{num_epochs}.csv'
            elif shadow_ring is False:
                f_params = f'./outputs/checkpoints/model_shadow_regions_{region}_{radius}_skeletonized_epoch{num_epochs}.pt'
                f_history = f'./outputs/histories/model_shadow_regions_{region}_{radius}_skeletonized_epoch{num_epochs}.json'
                csv_name = f'./outputs/probabilities/shadow_regions_{region}_{radius}_skeletonized_epoch{num_epochs}.csv'
        elif shadow is False:
            f_params = f'./outputs/checkpoints/model_skeletonized_epoch{num_epochs}.pt'
            f_history = f'./outputs/histories/model_skeletonized_epoch{num_epochs}.json'
            csv_name = f'./outputs/probabilities/skeletonized_epoch{num_epochs}.csv'
        
    elif shadow is True: # Experiments w/ skeleton false, shadow true
        if shadow_ring is True:
            if num_rings > 1: # accounting only for num_rings = 2 (radiuses only up to 2 are included)
                f_params = f'./outputs/checkpoints/model_shadow_rings_{region}_{ring_radiuses[0][0]}_{ring_radiuses[0][1]}_{ring_radiuses[1][0]}_{ring_radiuses[1][1]}_epoch{num_epochs}.pt'
                f_history = f'./outputs/histories/model_shadow_rings_{region}_{ring_radiuses[0][0]}_{ring_radiuses[0][1]}_{ring_radiuses[1][0]}_{ring_radiuses[1][1]}_epoch{num_epochs}.json'
                csv_name = f'./outputs/probabilities/shadow_rings_{region}_{ring_radiuses[0][0]}_{ring_radiuses[0][1]}_{ring_radiuses[1][0]}_{ring_radiuses[1][1]}_epoch{num_epochs}.csv'
            elif num_rings <= 1:
                f_params = f'./outputs/checkpoints/model_shadow_rings_{region}_{ring_radiuses[0]}_{ring_radiuses[1]}_epoch{num_epochs}.pt'
                f_history = f'./outputs/histories/model_shadow_rings_{region}_{ring_radiuses[0]}_{ring_radiuses[1]}_epoch{num_epochs}.json'
                csv_name = f'./outputs/probabilities/shadow_rings_{region}_{ring_radiuses[0]}_{ring_radiuses[1]}_epoch{num_epochs}.csv'
        elif shadow_ring is False:
            f_params = f'./outputs/checkpoints/model_shadow_regions_{region}_{radius}_epoch{num_epochs}.pt'
            f_history = f'./outputs/histories/model_shadow_regions_{region}_{radius}_epoch{num_epochs}.json'
            csv_name = f'./outputs/probabilities/shadow_regions_{region}_{radius}_epoch{num_epochs}.csv'
    else: # Original training: skeleton false, shadow false
        f_params = f'./outputs/checkpoints/model_original_epoch{num_epochs}.pt'
        f_history = f'./outputs/histories/model_original_epoch{num_epochs}.json'
        csv_name = f'./outputs/probabilities/original_epoch{num_epochs}.csv'
        
    # fix these transforms w/ new optic disk. Done!
    
    optic_disk_csv = "../../optic_disk/DeepROP/quality_assurance/QA.csv"
    QA_csv, checksum_dict = checksum(optic_disk_csv)    
    
    train_transforms = transforms.Compose([transforms.Lambda(lambda img: shadow_regions(img, skeleton, determine_image_center(img, image_size, QA_csv, checksum_dict), shadow, radius, shadow_ring, num_rings, ring_radiuses, region)), # image size pre-defined
                                           # transforms.Resize(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(25),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])]) # why this normalizing?
    
    test_transforms = transforms.Compose([transforms.Lambda(lambda img: shadow_regions(img, skeleton, determine_image_center(img, image_size, QA_csv, checksum_dict), shadow, radius, shadow_ring, num_rings, ring_radiuses, region)),
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
    print(f'Shadow: {shadow}')
    print(f'Shadow Radius: {radius}')
    print(f'Shadow_Ring: {shadow_ring}')
    print(f'Number of Rings: {num_rings}')
    print(f'Inner Radius/Ring: {ring_radiuses[0]}')
    print(f'Outer Radius/Ring: {ring_radiuses[1]}')
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
    if not os.path.isdir(os.path.join('outputs', 'probabilities')):
        os.makedirs(os.path.join('outputs', 'probabilities'))
    if not os.path.isdir(os.path.join('outputs', 'checkpoints')):
        os.makedirs(os.path.join('outputs', 'checkpoints'))
    if not os.path.isdir(os.path.join('outputs', 'histories')):
        os.makedirs(os.path.join('outputs', 'histories'))

    data_dir = os.path.join('dataset')

    # experiment 6 part 1
    
    train(data_dir, 0, 2, [[0,15],[75,90]], 'dark_center',skeleton=False, shadow = True, shadow_ring = True)
    train(data_dir, 0, 2, [[0,15],[75,90]], 'dark_background',skeleton=False, shadow = True, shadow_ring = True)

    train(data_dir, 0, 2, [[0,30],[60,90]], 'dark_center',skeleton=False, shadow = True, shadow_ring = True) 
    train(data_dir, 0, 2, [[0,30],[60,90]], 'dark_background',skeleton=False, shadow = True, shadow_ring = True) 

