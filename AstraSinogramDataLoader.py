'''
The dataloader that will be used to load 128x128 images and create their sinograms
'''
# imprt libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import radon, rescale, resize, iradon
import torchvision.transforms as T
from torchvision import datasets
import astra


dir_images = '/home/youness/data/Data/selected_data/Images_Png/'
tdir_images= '/home/youness/data/Data/selected_data/Images_Png_Test/'

dir_images = '/home/youness/data/Data/selected_data/Brain_PET_Png/'
tdir_images= '/home/youness/data/Data/selected_data/Brain_PET_Png_Test/'

def create_sinogram(image):
    angle_theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=angle_theta, circle=True, preserve_range=True) #, preserve_range=True
    return sinogram

def astra_create_sinogram(image, proj_id):
    _, sinogram = astra.create_sino(image, proj_id)
    return sinogram #.T


def add_noise_to_sino(sinogram_in, I0, seed=None):
    if seed is not None:
        curstate = np.random.get_state()
        np.random.seed(seed)

    sinogramRaw = sinogram_in
    max_sinogramRaw = sinogramRaw.max()
    sinogramRawScaled = sinogramRaw / max_sinogramRaw
    # to detector count
    sinogramCT = I0 * np.exp(-sinogramRawScaled)
    # add poison noise
    sinogramCT_C = np.zeros_like(sinogramCT)
    for i in range(sinogramCT_C.shape[0]):
        for j in range(sinogramCT_C.shape[1]):
            # Ensure lam is non-negative and not NaN
            lam = sinogramCT[i, j] #max(sinogramCT[i, j], 0)
            if np.isnan(lam):
                lam = 0  # Set to 0 or some other default value if NaN
            sinogramCT_C[i, j] = np.random.poisson(lam) 
    # to density
    sinogramCT_D = sinogramCT_C / I0
    sinogram_out = -max_sinogramRaw * np.log(sinogramCT_D + 1e-12)  # Add a small constant to avoid log(0)
    # make values < 0 to 0 in sinogram_out:
    #sinogram_out = np.where(sinogram_out<0, 0, sinogram_out)
    sinogram_out = np.abs(sinogram_out)
    if seed is not None:
        np.random.set_state(curstate)

    return sinogram_out

'''def astra_create_sinogram_w_noise(image, proj_id, io_value=1000, count_poucentage=1.0):
    _, sinogram = astra.create_sino(image, proj_id)
    print("sum of counts in the sinogram: ", np.sum(sinogram))
    sinogram = downsample_sinogram(sinogram, count_poucentage)
    print("sum of counts in the sinogram: ", np.sum(sinogram))
    # create a sinogram mask for the sinogram: for values > 0, set to 1 and for values <= 0, set to 0 with numpy:
    sinogram_mask = np.where(sinogram>0, 1, 0)

    sinogram = add_noise_to_sino(sinogram_in=sinogram, I0=io_value)
    
    # multiply the sinogram with the mask:
    sinogram = sinogram * sinogram_mask

    return sinogram #.T'''

def astra_create_sinogram_w_noise(image, proj_id, io_value=1000):
    _, sinogram = astra.create_sino(image, proj_id)
    # create a sinogram mask for the sinogram: for values > 0, set to 1 and for values <= 0, set to 0 with numpy:
    sinogram_mask = np.where(sinogram>0, 1, 0)
    sinogram = add_noise_to_sino(sinogram_in=sinogram, I0=io_value)
    
    # multiply the sinogram with the mask:
    sinogram = sinogram * sinogram_mask
    return sinogram #.T

class AstraSinogramImageDataset(Dataset):
    def __init__(self, image_folder, timage_folder, mode='train', num_img_train=100, num_img_test=100, new_shape=128, vol_geom=None, proj_geom=None, proj_id=None):
        # Store the paths to the sinogram and image arrays
        self.timage_paths = [os.path.join(timage_folder, f) for f in os.listdir(timage_folder)]
        self.image_paths  = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.timage_folder = timage_folder
        self.image_folder  = image_folder
        self.mode = mode
        self.num_img_train = num_img_train
        self.num_img_test  = num_img_test
        self.img_size = new_shape
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.proj_id = proj_id
    
    def __len__(self):
        if self.mode == 'train':
            return self.num_img_train
        else:
            return self.num_img_test
    
    def __getitem__(self, idx):
        # Load the sinogram and image arrays from the specified index
        image_path = self.image_paths[idx] if self.mode=='train' else self.timage_paths[idx]
        
        image    = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # reshape image to 64x64:
        image = cv2.resize(image, (self.img_size , self.img_size ))



        # Apply rotation (e.g., 360 degrees random rotation)
        angle = np.random.randint(0, 360)
        M = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1)
        image = cv2.warpAffine(image, M, (self.img_size, self.img_size))
        
        # Apply translation (e.g., random translation in both x and y directions)
        x_translation = np.random.randint(-10, 10)
        y_translation = np.random.randint(-10, 10)
        M = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
        image = cv2.warpAffine(image, M, (self.img_size, self.img_size))



        sinogram = astra_create_sinogram_w_noise(image, self.proj_id, io_value=500)
        image    = torch.from_numpy(image)
        sinogram = torch.from_numpy(sinogram)
        
        image = torch.unsqueeze(image, 0)
        sinogram = torch.unsqueeze(sinogram, 0)
        
        # Return the sinogram and image tensors as a tuple
        return sinogram, image


def downsample_sinogram(sinogram, count_poucentage=1.0):
    """
    Downsamples a PET sinogram by randomly selecting 1/4 of the events,
    maintaining the original event distribution.
    
    Parameters:
    - sinogram: 2D numpy array of the sinogram counts.
    
    Returns:
    - A 2D numpy array representing the downsampled sinogram.
    """
    # Flatten the sinogram to simplify the sampling process
    flat_sinogram = sinogram.flatten()
    
    # Calculate the total number of events to select (1/4 of the total events)
    total_events = np.sum(flat_sinogram)
    events_to_select = int(total_events * count_poucentage)
    
    # Create a probability distribution for sampling
    probabilities = flat_sinogram / total_events
    
    # Randomly select indices from the flattened sinogram based on the distribution
    selected_indices = np.random.choice(len(flat_sinogram), size=events_to_select, p=probabilities)
    
    # Reconstruct the downsampled sinogram
    downsampled_flat = np.zeros_like(flat_sinogram)
    np.add.at(downsampled_flat, selected_indices, 1)
    
    # Reshape back to the original sinogram shape
    downsampled_sinogram = downsampled_flat.reshape(sinogram.shape)
    
    return downsampled_sinogram

class AstraSinogramImageDataset_w_noise(Dataset):
    def __init__(self, image_folder, timage_folder, mode='train', num_img_train=100, num_img_test=100, new_shape=128, vol_geom=None, proj_geom=None, proj_id=None):
        # Store the paths to the sinogram and image arrays
        self.timage_paths = [os.path.join(timage_folder, f) for f in os.listdir(timage_folder)]
        self.image_paths  = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.timage_folder = timage_folder
        self.image_folder  = image_folder
        self.mode = mode
        self.num_img_train = num_img_train
        self.num_img_test  = num_img_test
        self.img_size = new_shape
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.proj_id = proj_id
    
    def __len__(self):
        if self.mode == 'train':
            return self.num_img_train
        else:
            return self.num_img_test
    
    def __getitem__(self, idx):
        # Load the sinogram and image arrays from the specified index
        image_path = self.image_paths[idx] if self.mode=='train' else self.timage_paths[idx]
        
        image    = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # reshape image to 64x64:
        image = cv2.resize(image, (self.img_size , self.img_size ))



        # Apply rotation (e.g., 360 degrees random rotation)
        angle = np.random.randint(0, 360)
        M = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1)
        image = cv2.warpAffine(image, M, (self.img_size, self.img_size))
        
        # Apply translation (e.g., random translation in both x and y directions)
        x_translation = np.random.randint(-10, 10)
        y_translation = np.random.randint(-10, 10)
        M = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
        image = cv2.warpAffine(image, M, (self.img_size, self.img_size))
        
        
        sinogram = astra_create_sinogram_w_noise(image, self.proj_id, io_value=1000)
        image    = torch.from_numpy(image)
        sinogram = torch.from_numpy(sinogram)
        
        image = torch.unsqueeze(image, 0)
        sinogram = torch.unsqueeze(sinogram, 0)
        
        # Return the sinogram and image tensors as a tuple
        return sinogram, image


'''def create_crisscross_pattern(num_angles, num_detectors, repetetion):
        # Initialize the sinogram with zeros (all detectors are "alive")
        sinogram = np.zeros((num_angles, num_detectors))

        # Set detectors to "dead" along both diagonal directions
        for i in range(num_angles):
            for j in range(num_detectors):
                if (i + j) % repetetion == 0 or (i - j) % repetetion == 0:
                    sinogram[i, j] = 1  # Set to "dead"

        return sinogram'''

def create_crisscross_pattern(num_angles, num_detectors, repetition):
    # Create grid of indices for angles and detectors
    angles, detectors = np.indices((num_angles, num_detectors))

    # Calculate the crisscross pattern without loops
    crisscross_mask = ((angles + detectors) % repetition == 0) | ((angles - detectors) % repetition == 0)

    # Create sinogram
    sinogram = np.zeros((num_angles, num_detectors), dtype=int)
    sinogram[crisscross_mask] = 1  # Set to "dead"

    return sinogram

def astra_create_sinogram_dead_region(image, proj_id, num_angles, num_detectors, io_value=1000, count_poucentage=1.0):
    _, sinogram = astra.create_sino(image, proj_id)
    if count_poucentage<1.0:
        sinogram = downsample_sinogram(sinogram, count_poucentage)
    # repetetion, should be a random number between 5 and 20
    repetition = np.random.randint(7, 20)
    dead_mask = create_crisscross_pattern(num_angles, num_detectors, repetition=repetition)

    sinogram = add_noise_to_sino(sinogram_in=sinogram, I0=io_value)
    sinogram = sinogram * (1-dead_mask)
    
    return sinogram, dead_mask  # No need to transpose unless required for further processing
class AstraSinogramImageDataset_w_DeadRegion(Dataset):
    def __init__(self, image_folder, timage_folder, mode='train', num_img_train=100, num_img_test=100, new_shape=128, vol_geom=None, proj_geom=None, proj_id=None, count_poucentage=1.0, poisson_noise=1000):
        # Store the paths to the sinogram and image arrays
        self.timage_paths = [os.path.join(timage_folder, f) for f in os.listdir(timage_folder)]
        self.image_paths  = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.timage_folder = timage_folder
        self.image_folder  = image_folder
        self.mode = mode
        self.num_img_train = num_img_train
        self.num_img_test  = num_img_test
        self.img_size = new_shape
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.proj_id = proj_id
        self.count_poucentage = count_poucentage
        self.poisson_noise = poisson_noise
    
    def __len__(self):
        if self.mode == 'train':
            return self.num_img_train
        else:
            return self.num_img_test
    
    def __getitem__(self, idx):
        # Load the sinogram and image arrays from the specified index
        image_path = self.image_paths[idx] if self.mode=='train' else self.timage_paths[idx]
        
        image    = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # reshape image to 64x64:
        image = cv2.resize(image, (self.img_size , self.img_size ))

        # Apply rotation (e.g., 360 degrees random rotation)
        angle = np.random.randint(0, 360)
        M = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1)
        image = cv2.warpAffine(image, M, (self.img_size, self.img_size))
        
        # Apply translation (e.g., random translation in both x and y directions)
        x_translation = np.random.randint(-10, 10)
        y_translation = np.random.randint(-10, 10)
        M = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
        image = cv2.warpAffine(image, M, (self.img_size, self.img_size))
        

        sinogram, dead_region_mask = astra_create_sinogram_dead_region(image, self.proj_id, self.img_size, self.img_size, io_value=self.poisson_noise, count_poucentage=self.count_poucentage)
        image    = torch.from_numpy(image)
        sinogram = torch.from_numpy(sinogram)
        dead_region_mask = torch.from_numpy(dead_region_mask)
        
        image = torch.unsqueeze(image, 0)
        sinogram = torch.unsqueeze(sinogram, 0)
        dead_region_mask = torch.unsqueeze(dead_region_mask, 0)
        
        # Return the sinogram and image tensors as a tuple
        return sinogram, image, dead_region_mask




if __name__ == '__main__':
    dataset_root = "/home/youness/data/Perso_Learning/guide_for_3-d_-image_-reconstruction_with_-ct_-pet-master/mnist_data/"

    img_size = 128
    num_pixels = 128
    num_detectors = 128
    detector_size = 1
    num_angles = 180 #180
    vol_geom  = astra.create_vol_geom(img_size, img_size)
    proj_geom = astra.create_proj_geom('parallel', detector_size, num_detectors, np.linspace(0,np.pi,num_angles,False))
    proj_id   = astra.create_projector('strip', proj_geom, vol_geom)
    

    RecoDataset = AstraSinogramImageDataset_w_DeadRegion(dir_images, tdir_images, mode='train', vol_geom=vol_geom, proj_geom=proj_geom, proj_id=proj_id)
    show_dataloader = DataLoader(RecoDataset, batch_size=4, shuffle=False)
    
    

   


    for sinograms, images, DR_sino in show_dataloader:
        # 'sinograms' contains the sinogram data (batch)
        # 'images' contains the corresponding MNIST images (batch)
        # plot 4 examples from the batch:
        for i in range(4):
            plt.subplot(2, 4, i+1)
            plt.imshow(sinograms[i,0], cmap='gray')
            plt.subplot(2, 4, i+5)
            plt.imshow(images[i,0], cmap='gray')

            
        plt.show()
   