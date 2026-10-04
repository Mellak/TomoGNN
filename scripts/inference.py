"""Run a trained TomoGNN on simulated noisy sinograms and plot sinogram / ground truth / reconstruction / difference.

Examples:
    # random images from a folder of PNGs (opens matplotlib windows)
    python scripts/inference.py --test-dir Images/BrainWeb --num-img-test 34

    # one image stored as a .npy slice, e.g. the sample shipped in this repo
    python scripts/inference.py --single-image Images/PET_test_slice_452.npy
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from tomognn.data import AstraSinogramImageDataset_w_noise
from tomognn.models import TomoGNN
from tomognn.utils import build_astra_graph, get_image_and_sinogram

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / 'weights' / 'paradigm2_weights_epoch_233.pth'


def Inference_on_single_image(global_model, weights, image_path, num_pixels, img_size, edge_index, edge_weight, device):
    # Load the model:
    state_dict = torch.load(weights, map_location=device)  # map_location: the shipped weights were saved from a GPU
    global_model.load_state_dict(state_dict)

    # import an image and get the sinogram:
    image = np.load(image_path)
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

def Inference(global_model, test_dataloader, weights, num_pixels, edge_index, edge_weight, device):
    # Load the model:
    state_dict = torch.load(weights, map_location=device)  # map_location: the shipped weights were saved from a GPU
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', default=str(DEFAULT_WEIGHTS), help='state_dict (.pth) to load (default: %(default)s)')
    p.add_argument('--single-image', metavar='NPY', help='reconstruct this single .npy image instead of sampling from --test-dir')
    p.add_argument('--test-dir', help='folder of PNG images to sample from (required unless --single-image is given)')
    p.add_argument('--num-pixels', type=int, default=128, help='image size = number of detectors = number of angles (default: %(default)s)')
    p.add_argument('--num-img-test', type=int, default=800, help='number of images drawn from --test-dir; must not exceed the number of files in it (default: %(default)s)')
    args = p.parse_args(argv)
    if args.single_image is None and args.test_dir is None:
        p.error('either --test-dir or --single-image is required')
    return args


def lauch_inference(argv=None):
    args = parse_args(argv)
    num_pixels = img_size = num_detectors = num_angles = args.num_pixels

    vol_geom, proj_geom, proj_id, edge_index, edge_weight = build_astra_graph(
        num_pixels=num_pixels, num_detectors=num_detectors, num_angles=num_angles, detector_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu')
    global_model = TomoGNN().to(device)

    if args.single_image is not None:
        Inference_on_single_image(global_model, args.weights, args.single_image, num_pixels, img_size, edge_index, edge_weight, device)
    else:
        RecoDataset = AstraSinogramImageDataset_w_noise(args.test_dir, args.test_dir, mode='test', num_img_train=50, num_img_test=args.num_img_test, vol_geom=vol_geom, proj_geom=proj_geom, proj_id=proj_id, new_shape=num_pixels)
        test_dataloader = DataLoader(RecoDataset, batch_size=1, shuffle=True)
        Inference(global_model, test_dataloader, args.weights, num_pixels, edge_index, edge_weight, device)


if __name__ == '__main__':
    lauch_inference()
