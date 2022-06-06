"""
Author: Praveer SIngh
Date: 04/04/2021

Model: pre-trained ResNet18
Dataset: Training on Train/Val/Test splits from BPD+Full_ROP dataset from Pete
oversampling of BPD class
originally trained using MONAI version: 0.3.0rc4
"""
import os
import pandas as pd
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision.models

from torchvision.transforms import Lambda, Normalize 

from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import compute_roc_auc
from monai.transforms import (
    Activations,
    AddChannel,
    AsChannelFirst,
    Compose,
    LoadPNG,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity, Transpose, 
    LoadImage,
    ToTensor,
    Resize,
)
from monai.utils import set_determinism
from dataset import ROPDatasetRGBImage
from utils import BPD_CNN, train, val, test

print("Environment onfiguration is:")
print_config()
new_datadir = "./../Posterior"

dict_label2idx = {0.0:int(0),1.0:int(1)}


df_test_plus_unclean = pd.read_csv('test.csv', usecols=['imagename','o236'])
#df_test = df_test.replace({'o236': dict_label2idx})
df_test_plus_unclean['o236'] = df_test_plus_unclean['o236'].astype(int)
df_test_plus_unclean = df_test_plus_unclean.rename(columns={"imagename": "imageName", "o236": "label"})

class_count_test_plus_unclean = np.unique(df_test_plus_unclean['label'], return_counts=True)[1]

test_plus_unclean_paths_old = df_test_plus_unclean['imageName'].tolist() 
test_plus_unclean_image_files_list = [os.path.join(new_datadir, i) for i in test_plus_unclean_paths_old] # combine the image name with the new datapath 

list_all_images = []


test_plus_unclean_image_class = df_test_plus_unclean['label'].tolist()
test_plus_unclean_image_class_list = []

for elem in os.listdir('./../Posterior/'):
    list_all_images.append(new_datadir + '/' + elem)

test_plus_unclean_image_files_list_updated = []


for i, elem in enumerate(test_plus_unclean_image_files_list):
    if elem in list_all_images:
        test_plus_unclean_image_files_list_updated.append(elem)
        test_plus_unclean_image_class_list.append(test_plus_unclean_image_class[i])

"""
transforms 
"""
test_transforms = Compose(
    [
        ScaleIntensity(minv=0.0,maxv=1.0),
        AsChannelFirst(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
# Dataloader
batch_size = 64
test_plus_unclean_ds = ROPDatasetRGBImage(test_plus_unclean_image_files_list_updated, test_plus_unclean_image_class_list, test_transforms)
test_plus_unclean_loader = torch.utils.data.DataLoader(test_plus_unclean_ds, batch_size=batch_size, num_workers=4)

# Model
model = BPD_CNN().cuda()
device = torch.device("cuda:0")
model.to(device)


model_dir = "./models_resnet18_bs64_lr1e-4_epoch8_rawimage_size480X640_dataset_normalized_without_CLAHE_normalized_PLUS_CLEANED/"
model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth")))

features = {}
act = Activations(sigmoid=True)

# Extracting features
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


##### FEATURE EXTRACTION

def test(model, test_loader, device, model_dir):
    
    model.eval()
    model.resnet18.avgpool.register_forward_hook(get_features('feats'))
    with torch.no_grad():
    
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y_feats = torch.tensor([],dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)

        test_filenames_all = []
        for test_data in test_loader:
            test_images, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
            )
                
            test_filenames = test_data[2]
            test_filenames_all.extend(test_filenames)

            y_pred = torch.cat([y_pred, model(test_images)], dim=0)
            y_feats = torch.cat([y_feats, features['feats']],dim=0)
            y = torch.cat([y, test_labels], dim=0)
        
        df_out = pd.DataFrame(data=y_feats.squeeze().cpu().numpy(), index=test_filenames_all)
        df_out['filename'] = test_filenames_all
        df_out['label'] = y.cpu().numpy()
        filename = os.path.join(model_dir, 'BPD_test_unclean_features.csv')
        df_out.to_csv(filename,index=False)



test(model, test_plus_unclean_loader, device, model_dir)
