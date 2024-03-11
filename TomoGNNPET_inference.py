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
from models import TomoGNN



def MLEM_reconstruct(sinogram, proj_id, iterations):
    # Initialize the OpTomo object with the given projector ID
    W = astra.optomo.OpTomo(proj_id)
    
    # Initialize the reconstruction volume 'x' with ones, assuming 'sinogram' has the correct shape
    x_shape = (sinogram.shape[1], sinogram.shape[0]) # This might need adjustment based on your setup
    x = np.ones(x_shape, dtype=np.float32)
    
    for n in range(iterations):
        # Forward projection of the current estimate 'x'
        yP = np.empty_like(sinogram, dtype=np.float32)
        W.FP(x, out=yP)
        
        # Compute the correction factor from the ratio of the measured projections to the estimated projections
        yN = sinogram / yP
        
        # Backprojection of the correction factors
        xR = np.empty_like(x, dtype=np.float32)
        W.BP(yN, out=xR)
        
        # Update the image estimate by element-wise multiplication with the correction factors
        x *= xR
        
        # Normalize the update step (optional, depending on your normalization strategy)
        # This could be an additional backprojection of ones and division by this result
        normalization_factor = np.empty_like(x, dtype=np.float32)
        W.BP(np.ones_like(sinogram, dtype=np.float32), out=normalization_factor)
        x /= normalization_factor
    
    # Normalize the final image (optional, for visualization purposes)
    #x = (x - x.min()) / (x.max() - x.min())
    
    return x

def Rec_FBP(sinogram,proj_id,vol_geom,proj_geom):
    
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    cfg['option'] = {}
    cfg['option']['FilterType'] = 'hamming'

    alg_id = astra.algorithm.create(cfg)
    #time.sleep(0.00000001)#if not kernel dies
    astra.algorithm.run(alg_id)
    #time.sleep(0.00000001)
    x = astra.data2d.get(rec_id)
    #x = (x - x.min())/ (x.max() - x.min())
    
    return x

def get_image_and_sinogram(image, num_pixels):
    img_size = num_pixels
    num_detectors = num_pixels
    detector_size = 1
    num_angles = num_pixels #180
    vol_geom  = astra.create_vol_geom(img_size, img_size)
    proj_geom = astra.create_proj_geom('parallel', detector_size, num_detectors, np.linspace(0,np.pi,num_angles,False))
    proj_id   = astra.create_projector('strip', proj_geom, vol_geom)

    image = cv2.resize(image, (img_size , img_size ))

    sinogram = astra_create_sinogram(image, proj_id)
    sinogram_noisy = astra_create_sinogram_w_noise(image, proj_id, io_value=1000)
    return image, sinogram, sinogram_noisy


# Define hyperparameters:
num_pixels    = 128 #256
img_size      = 128 #256
num_detectors = 128 #256
detector_size = 1
num_angles    = 128

vol_geom  = astra.create_vol_geom(num_pixels, num_pixels)
proj_geom = astra.create_proj_geom('parallel', detector_size, num_detectors, np.linspace(0,np.pi,num_angles,False))
proj_id   = astra.create_projector('strip', proj_geom, vol_geom)

system_matrix_id = astra.projector.matrix(proj_id)
system_matrix    = astra.matrix.get(system_matrix_id)
adj_matrix       = sp.csr_matrix(system_matrix)
# get tensor from adj_matrix:
edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
edge_index = torch.tensor(edge_index.clone().detach())
edge_index[1] = edge_index[1] + torch.max(edge_index[0]) + 1
edge_weight = torch.tensor(edge_weight.clone().detach())



RecoDataset = AstraSinogramImageDataset_w_noise(dir_images, tdir_images, mode='test', num_img_train=50, num_img_test=800, vol_geom=vol_geom, proj_geom=proj_geom, proj_id=proj_id, new_shape=num_pixels)
test_dataloader = DataLoader(RecoDataset, batch_size=1, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu')


def Inference_on_single_image(global_model):
    # Load the model:
    state_dict = torch.load('/home/youness/data/PET_Recons/AstraGNNMICCAI/MyModel/Weights4test/'+"paradigm2_weights_epoch_233.pth")
    global_model.load_state_dict(state_dict)

    # import an image and get the sinogram:
    img_number = "452"
    image = np.load("/home/youness/data/Data/TBodyPET/Images_test_Png/PET_test_slice_"+str(img_number)+".npy") #404 #300 # 500 # 561 #351
    image = cv2.resize(image.squeeze(), (img_size,img_size))
    _, sino, noisy_sino = get_image_and_sinogram(image, num_pixels)


    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    noisy_sino = torch.from_numpy(noisy_sino).unsqueeze(0).unsqueeze(0).to(device)
    
                
    my_img  = image.float() #torch.zeros_like(image) #image.float() #
    my_sino = noisy_sino.float()
    
    out = global_model(my_sino, edge_index, edge_weight)
    
    # select nodes of the image space, then normalize the output:
    my_out = out[-num_pixels*num_pixels:].reshape(num_pixels,num_pixels).cpu().detach().numpy()
    my_out = (my_out - np.min(my_out)) / (np.max(my_out) - np.min(my_out))
    # normalizing the input:
    my_img = my_img.squeeze().cpu().detach().numpy()
    my_img = (my_img - np.min(my_img)) / (np.max(my_img) - np.min(my_img))

    fig, axs = plt.subplots(1,4)
    max_cmap = torch.max(my_sino).item() #/ 0.05
    axs[0].imshow(my_sino.squeeze().cpu().detach().numpy(), cmap='gray', vmin=0, vmax=max_cmap)
    axs[0].set_title('Sinogram')

    
    axs[1].imshow(my_img, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Image')
    
    # normalizing the output:
    axs[2].imshow(my_out, cmap='gray', vmin=0, vmax=1)
    axs[2].set_title('Projection2ImageSpace')
    
    axs[3].imshow(my_img - my_out, cmap='seismic', vmin=-1, vmax=1)
    axs[3].set_title('Diff')
    plt.show()

def Inference(global_model, test_dataloader):
    # Load the model:
    state_dict = torch.load('/home/youness/data/PET_Recons/AstraGNNMICCAI/MyModel/Weights4test/'+"paradigm2_weights_epoch_233.pth")
    global_model.load_state_dict(state_dict)

    with torch.no_grad():
        for sinograms, images in test_dataloader:
            for batch in range(sinograms.shape[0]):
                my_sino = sinograms.float().to(device) 
                my_img  = images.float().to(device)

                
                out = global_model(my_sino, edge_index, edge_weight)
                
                # select nodes of the image space, then normalize the output:
                my_out = out[-num_pixels*num_pixels:].reshape(num_pixels,num_pixels).cpu().detach().numpy()
                my_out = (my_out - np.min(my_out)) / (np.max(my_out) - np.min(my_out))
                # normalizing the input:
                my_img = my_img.squeeze().cpu().detach().numpy()
                my_img = (my_img - np.min(my_img)) / (np.max(my_img) - np.min(my_img))

                fig, axs = plt.subplots(1,4)
                max_cmap = torch.max(my_sino).item() #/ 0.05
                axs[0].imshow(my_sino.squeeze().cpu().detach().numpy(), cmap='gray', vmin=0, vmax=max_cmap)
                axs[0].set_title('Sinogram')

                
                axs[1].imshow(my_img, cmap='gray', vmin=0, vmax=1)
                axs[1].set_title('Image')
                
                # normalizing the output:
                axs[2].imshow(my_out, cmap='gray', vmin=0, vmax=1)
                axs[2].set_title('Projection2ImageSpace')
               
                axs[3].imshow(my_img - my_out, cmap='seismic', vmin=-1, vmax=1)
                axs[3].set_title('Diff')
                plt.show()



def lauch_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu')
    global_model = TomoGNN().to(device)
    Inference(global_model, test_dataloader)
    #Inference_on_single_image(global_model)

if __name__ == '__main__':
    lauch_inference()