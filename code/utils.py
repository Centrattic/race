import os
import pandas as pd
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch import nn
import torchvision.models
import cv2

from torchvision.transforms import Lambda, Normalize 

from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import compute_roc_auc
from torchvision import models
from torchvision.models import resnet18
from monai.transforms import (
    Activations,
    AddChannel,
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
set_determinism(seed=0)

act = Activations(sigmoid=True)

class Race_CNN(nn.Module): 
    def __init__(self):
        
        super(Race_CNN, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = torch.nn.Linear(num_ftrs, 1)

    def forward(self, x):
        output = self.resnet18(x)
        output = torch.squeeze(output, dim = -1)
        return output

def train(epoch, model, criterion, optimizer, train_loader, device, model_dir):
    total_correct = 0
    metric_values = list()
    epoch_loss_values = list()
    model.train()
    epoch_loss = 0
    step = 0
    num_elements = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        num_elements += inputs.shape[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        total_correct += torch.eq(act(outputs).round(), labels).sum().item()
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    
    acc_metric = float(total_correct) / num_elements
    print(
            f"Current Epoch: {epoch + 1} Train Accuracy:{acc_metric: .4f} Train Avg. Loss: {epoch_loss:.4f}"
    )

    torch.save(model.state_dict(), os.path.join(model_dir, "epoch_"+str(epoch+1)+".pth"))
    print("Saving model after Epoch ", epoch+1)


def val(epoch, model, val_loader, device, best_metric_val, best_metric_epoch, model_dir):
    metric_values = list()
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        val_filenames_all = []
        for val_data in val_loader:
            val_images, val_labels = (
                val_data[0].to(device),
                val_data[1].to(device),
            )
                
            val_filenames = val_data[2]
            val_filenames_all.extend(val_filenames)

            y_pred = torch.cat([y_pred, model(val_images)], dim=0)
            y = torch.cat([y, val_labels], dim=0)
        
        auc_metric = compute_roc_auc(act(y_pred), y)
        metric_values.append(auc_metric)
        
        acc_value = torch.eq(act(y_pred).round(), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        print(
                f"Current Epoch: {epoch + 1} Val accuracy: {acc_metric:.4f}  Val AUC: {auc_metric: .4f}"  
        )
        if auc_metric > best_metric_val:
            best_metric_val = auc_metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
            print("Saving New Best Metric model")
             
        print(f" Best Val AUC: {best_metric_val:.4f} at Epoch: {best_metric_epoch}")
    return best_metric_val, best_metric_epoch

def test(model, test_loader, device, model_dir):
    metric_values = list()
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
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
            y = torch.cat([y, test_labels], dim=0)
        
        auc_metric = compute_roc_auc(act(y_pred), y)
        metric_values.append(auc_metric)
        
        df_out = pd.DataFrame({'filename':test_filenames_all, 'predictions': act(y_pred).cpu().numpy()})
        filename = os.path.join(model_dir, 'Race_classification.csv')
        df_out.to_csv(filename,index=False)
        acc_value = torch.eq(act(y_pred).round(), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        print(
            f"Test Accuracy: {acc_metric:.4f} Test AUC: {auc_metric:.4f}"
        )

        best_metric_test=auc_metric
    
    return best_metric_test
