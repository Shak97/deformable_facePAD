import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
import torch.optim as optim
import os
import random
import cv2
import numpy as np
import glob
from PIL import Image


from deformable_conv import DeformableConv2d

device = torch.device(0)

class_label_real = 0
class_label_attack = 1

model = models.mobilenet_v2(pretrained=True)
# model.features[0] = DeformableConv2d(3, 32, 3, 2, 1)
model.features[-1] = nn.Sequential(
    DeformableConv2d(320, 1280, 3, 2),
    nn.BatchNorm2d(1280),
    nn.ReLU6()
)
model.classifier[1] = nn.Linear(in_features=1280, out_features=2) #default in_features =1280, out_features = 1000

best_model_path = r'checkpoint_protocol_1_wo_val/best_142_4.0373697962606216e-07.pth'
state_dict = torch.load(best_model_path,  map_location=device)
model.load_state_dict(state_dict, strict = True)
model = model.to(device)

input = torch.rand((16,3,224,224), dtype = torch.float32)

output = model(input.to(device))
