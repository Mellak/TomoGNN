from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale, resize, iradon
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm  # Import tqdm
import astra
import cv2
import os
from AstraSinogramDataLoader import *
import math
from torch_geometric.nn.conv import GCNConv



def build_nodes_features(sinogram=None, image=None, flow='source_to_target', num_pixels=128, num_detectors=128, num_angles=128):
    # check if image not None, then get batch_size, num_channels, num_pixels, num_pixels
    if image is not None:
        batch_size, num_features, _, _ = image.shape
        my_device = image.device
    # check if sinogram not None, then get batch_size, num_channels, num_detectors, num_angles
    if sinogram is not None:
        batch_size, num_features, _, _ = sinogram.shape
        my_device = sinogram.device

    if flow == 'source_to_target':
        # change the channels to be the last dimension:
        sinogram = sinogram.permute(0, 2, 3, 1)
        flatten_sino = sinogram.reshape(batch_size, num_detectors*num_angles, num_features)
        flatten_img  = torch.zeros((batch_size, num_pixels*num_pixels, num_features)).to(my_device)
    if flow == 'target_to_source':
        # make the channels to be the last dimension:
        image = image.permute(0, 2, 3, 1)
        flatten_img = image.reshape(batch_size, num_pixels*num_pixels, num_features)
        flatten_sino  = torch.zeros((batch_size, num_detectors*num_angles, num_features)).to(my_device)

    nodes_matrix = torch.cat((flatten_sino, flatten_img), dim=1)
    
    return nodes_matrix

class MessagePassBackproject(MessagePassing):
    def __init__(self, aggr='sum', flow='target_to_source'):
        super(MessagePassBackproject, self).__init__(aggr=aggr, flow=flow)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flow = flow

    def forward(self, input_data, edge_index, edge_weight):
        if self.flow == 'target_to_source':
            x = build_nodes_features(sinogram=None, image=input_data, flow='target_to_source', num_detectors=num_detectors, num_angles=num_angles, num_pixels=num_pixels).to(self.device)
        else:
            x = build_nodes_features(sinogram=input_data, image=None, flow='source_to_target', num_detectors=num_detectors, num_angles=num_angles, num_pixels=num_pixels).to(self.device)
        x, edge_index, edge_weight = x.to(self.device), edge_index.to(self.device), edge_weight.to(self.device)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        if self.flow == 'target_to_source':
            # predicting sino
            out = out[:,:num_detectors*num_angles].reshape(-1, 1, num_detectors,num_angles)
        else:
            # predicting image
            out = out[:,num_detectors*num_angles:].reshape(-1, 1, num_pixels, num_pixels)
        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

class ResidualCNN(nn.Module):
    def __init__(self):
        super(ResidualCNN, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, padding_mode='zeros'), nn.PReLU(),
                                   nn.Conv2d(8, 8, 3, padding=1, padding_mode='zeros'), nn.PReLU(),
                                   nn.Conv2d(8, 1, 1, padding=0), nn.PReLU(),)
    def forward(self, x):
        out = self.model(x)
        output = out + x
        return output
        
class SRNET(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, in_between_channel=64):
        super(SRNET, self).__init__()
        self.model = nn.Sequential(
          nn.Conv2d(in_channels=input_channel, out_channels=in_between_channel*2, kernel_size=9, stride=1, padding=9//2),
          nn.PReLU(),
          nn.Conv2d(in_channels=in_between_channel*2, out_channels=in_between_channel, kernel_size=1, stride=1, padding=0//2),
          nn.PReLU(),
          nn.Conv2d(in_channels=in_between_channel, out_channels=output_channel, kernel_size=5, stride=1, padding=5//2),
          nn.PReLU(),
        )

    def forward(self, input):
        out = self.model(input) #torch.clamp(self.model(input), min=1e-12, max=1-(1e-12))
        #out = out + input
        return out #torch.exp(out)
    
class AndrewCNN(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, in_between_channel=64):
        super(AndrewCNN, self).__init__()
        self.model = nn.Sequential(
        nn.Conv2d(input_channel, in_between_channel, 7, padding=(3,3)), nn.PReLU(),
        nn.BatchNorm2d(in_between_channel),
        nn.Conv2d(in_between_channel, in_between_channel, 7, padding=(3,3)), nn.PReLU(),
        nn.Conv2d(in_between_channel, in_between_channel, 7, padding=(3,3)), nn.PReLU(),
        nn.Conv2d(in_between_channel, in_between_channel, 7, padding=(3,3)), nn.PReLU(),
        nn.Conv2d(in_between_channel, output_channel, 7, padding=(3,3)), nn.PReLU(),
        )
    def forward(self, x):
        out = self.model(x)
        #output = out + x
        return out

class TomoGNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, num_pixels=128, num_detectors=128, num_angles=128):
        super().__init__()  # "Add" aggregation (Step 5).
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu')
        self.features_size = out_channels
        self.SR_model = SRNET(input_channel=self.features_size, output_channel=1, in_between_channel=32).to(self.device)
        self.gnn_conv = GCNConv(8, out_channels, bias=True, add_self_loops=True, normalize=True).to(self.device)
        self.sinogram_filter = AndrewCNN(input_channel=1, output_channel=8, in_between_channel=32).to(self.device)
        self.num_pixels = num_pixels
        self.num_detectors = num_detectors
        self.num_angles = num_angles

    def forward(self, sino, edge_index, edge_weight=None):
        
        sino = self.sinogram_filter(sino)
        #sino = sino.view(sino.shape[0], num_detectors*num_angles, 1).to(self.device)
        x = build_nodes_features(sino, num_detectors=self.num_detectors, num_angles=self.num_angles, num_pixels=self.num_pixels).to(self.device)
        
        x, edge_index, edge_weight = x.to(self.device), edge_index.to(self.device), edge_weight.float().to(self.device)

        # Step 2: Linearly transform node feature matrix.
        out = self.gnn_conv(x, edge_index, edge_weight)
        
        #SuperResoStuff: # selecting only features of image space:
        out = out[:,self.num_detectors*self.num_angles:].permute(0, 2, 1).reshape(-1, self.features_size, self.num_pixels, self.num_pixels)
        out = self.SR_model(out)
        
        return out
    
class GradientDifferenceLoss(nn.Module):
    def __init__(self):
        super(GradientDifferenceLoss, self).__init__()

    def forward(self, inputs, targets):
        # Calculate gradients along height (axis 2)
        gradient_diff_h = (inputs.diff(dim=-2) - targets.diff(dim=-2)).pow(2)
        
        # Calculate gradients along width (axis 3)
        gradient_diff_w = (inputs.diff(dim=-1) - targets.diff(dim=-1)).pow(2)
        
        # Sum up the gradient differences and average over all elements
        loss_gdl = (gradient_diff_h.sum() + gradient_diff_w.sum()) / inputs.numel()
        
        return loss_gdl