"""Train TomoGNN on sinograms simulated from a folder of PNG images.

Example:
    python scripts/train.py --train-dir data/Brain_PET_Png --test-dir data/Brain_PET_Png_Test --output-dir outputs/run1

Checkpoints go to <output-dir>/weights/TomoGCN.pth (training resumes from it if present);
a reconstruction preview per epoch goes to <output-dir>/images/.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')  # figures are only saved to disk
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from tomognn.data import AstraSinogramImageDataset_w_noise
from tomognn.models import TomoGNN, GradientDifferenceLoss
from tomognn.utils import build_astra_graph


def save_checkpoint(path, epoch, model, optimizer_Model):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_Model_state_dict': optimizer_Model.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer_Model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_Model.load_state_dict(checkpoint['optimizer_Model_state_dict'])
    return checkpoint['epoch']


def global_training_loop(model, dataloader, test_dataloader, optimizer, criterion, device,
                         edge_index, edge_weight, checkpoint_path, images_path, epochs):
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device) + 1

    for epoch in range(start_epoch, epochs):
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
        save_checkpoint(checkpoint_path, epoch, model, optimizer)

        evaluate_model(model, test_dataloader, epoch, device, edge_index, edge_weight, images_path)


def evaluate_model(model, dataloader, epoch, device, edge_index, edge_weight, images_path):
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train-dir', required=True, help='folder of training PNG images')
    p.add_argument('--test-dir', required=True, help='folder of test PNG images')
    p.add_argument('--output-dir', default='outputs', help='where checkpoints and previews are written (default: %(default)s)')
    p.add_argument('--num-pixels', type=int, default=128, help='image size = number of detectors = number of angles (default: %(default)s)')
    p.add_argument('--epochs', type=int, default=20000, help='last epoch + 1 (default: %(default)s, the bound hard-coded in the original script)')
    p.add_argument('--batch-size', type=int, default=5, help='training batch size (default: %(default)s)')
    p.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate (default: %(default)s)')
    p.add_argument('--num-img-train', type=int, default=50, help='(default: %(default)s)')
    p.add_argument('--num-img-test', type=int, default=10, help='(default: %(default)s)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    num_pixels = img_size = num_detectors = num_angles = args.num_pixels

    vol_geom, proj_geom, proj_id, edge_index, edge_weight = build_astra_graph(
        num_pixels=num_pixels, num_detectors=num_detectors, num_angles=num_angles, detector_size=1)

    path_Weights = os.path.join(args.output_dir, 'weights')
    images_path = os.path.join(args.output_dir, 'images')
    os.makedirs(path_Weights, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TomoGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # NOTE: kept as in the original script, the "training" dataset is built with mode='test',
    # i.e. it samples images from --test-dir (--train-dir is listed but not used for sampling).
    train_dataset = AstraSinogramImageDataset_w_noise(args.train_dir, args.test_dir, mode='test', num_img_train=args.num_img_train, num_img_test=args.num_img_test, vol_geom=vol_geom, proj_geom=proj_geom, proj_id=proj_id, new_shape=num_pixels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = AstraSinogramImageDataset_w_noise(args.train_dir, args.test_dir, mode='test', num_img_train=args.num_img_train, num_img_test=args.num_img_test, vol_geom=vol_geom, proj_geom=proj_geom, proj_id=proj_id, new_shape=img_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    global_training_loop(model, train_dataloader, test_dataloader, optimizer, criterion, device,
                         edge_index, edge_weight, os.path.join(path_Weights, 'TomoGCN.pth'), images_path, args.epochs)


if __name__ == '__main__':
    main()
