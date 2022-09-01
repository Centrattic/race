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

from utils_train import systemic_brightening

def random_systemic_brightening(img, skeleton, thresh_type = 'below'): # code not fully functioning for thresh_type = 'above'
    
    intensity_change = np.random.choice(['brighten', 'dull'])
    
    if skeleton == True:
        intensity_change = 'dull' # can't brighten more than 255
    
    brighten_sum = np.random.choice(150) # randomly a value from brighten_sum
    
    final_img = systemic_brightening(img, skeleton, thresh_type, intensity_change, brighten_sum)
    
    return final_img
    
    