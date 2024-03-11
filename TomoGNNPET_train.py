import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import astra
from AstraSinogramDataLoader import AstraSinogramImageDataset_w_noise
from models import TomoGNN, MessagePassBackproject, GradientDifferenceLoss, build_nodes_features
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import scipy.sparse as sp

# define expiriment parameters:
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



# Define paths:
#---data paths---
dir_images = '/home/youness/data/Data/selected_data/Brain_PET_Png/'
tdir_images= '/home/youness/data/Data/selected_data/Brain_PET_Png_Test/'
#---save paths---
experiment_path = '/home/youness/data/PET_Recons/AstraToolBoxwGNN/Data/MICCAI/'
path_Weights = experiment_path+'Weights_'+str(num_pixels)+'_smallTomoGCN_PET_w_noise_brain_paradigm2_v3/'
images_path  = experiment_path+'Images_'+str(num_pixels)+'x'+str(num_pixels)+'_smallTomoGCN_PET_w_noise_brain_paradigm2_v3/'
if not os.path.exists(path_Weights):
    os.makedirs(path_Weights)
if not os.path.exists(images_path):
    os.makedirs(images_path)

def save_checkpoint(epoch, model, optimizer_Model):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_Model_state_dict': optimizer_Model.state_dict(),
    }
    torch.save(state, path_Weights+"TomoGCN.pth")

def load_checkpoint(model, optimizer_Model):
    checkpoint = torch.load(path_Weights+"TomoGCN.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_Model.load_state_dict(checkpoint['optimizer_Model_state_dict'])
    return checkpoint['epoch']


def global_training_loop(model, dataloader, test_dataloader, optimizer, criterion, device):
    start_epoch = 0
    if os.path.exists(os.path.join(path_Weights, "TomoGCN.pth")):
        start_epoch = load_checkpoint(model, optimizer) + 1

    for epoch in range(start_epoch, 20000):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}, Loss: {total_loss:.4f}', leave=False)

        for sinograms, images in progress_bar:
            sinograms, images = sinograms.float().to(device), images.float().to(device)
             
            optimizer.zero_grad()
            outputs = model(sinograms, edge_index, edge_weight)

            loss1 = criterion(outputs, images)
            loss2 = GradientDifferenceLoss()(outputs, images)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch}, Loss: {loss:.4f}')

        print(f'Epoch {epoch}, Loss {total_loss / len(dataloader)}')
        save_checkpoint(epoch, model, optimizer)

        if epoch % 1 == 0:
            evaluate_model(model, test_dataloader, epoch, device)

def evaluate_model(model, dataloader, epoch, device):
    model.eval()
    with torch.no_grad():
        for sinograms, images in dataloader:
            sinograms, images = sinograms.float().to(device), images.float().to(device)
            outputs = model(sinograms, edge_index, edge_weight)
            break  # Only evaluate on the first batch

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(images.cpu().squeeze(), cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(outputs.cpu().squeeze(), cmap='gray')
        axs[1].set_title('Reconstructed Image')
        plt.savefig(os.path.join(images_path, f'epoch_{epoch}.png'))
        plt.close()
    model.train()



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TomoGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()


    train_dataset = AstraSinogramImageDataset_w_noise(dir_images, tdir_images, mode='test', num_img_train=50, num_img_test=10, vol_geom=vol_geom, proj_geom=proj_geom, proj_id=proj_id, new_shape=num_pixels)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    test_dataset = AstraSinogramImageDataset_w_noise(dir_images, tdir_images, mode='test', num_img_train=50, num_img_test=10, vol_geom=vol_geom, proj_geom=proj_geom, proj_id=proj_id, new_shape=img_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    global_training_loop(model, train_dataloader, test_dataloader, optimizer, criterion, device)

if __name__ == '__main__':
    main()
