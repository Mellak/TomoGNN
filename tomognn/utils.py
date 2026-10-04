"""Shared helpers: ASTRA graph construction and reference reconstructions (MLEM / FBP)."""
import cv2
import numpy as np
import scipy.sparse as sp
import torch
import astra
from torch_geometric.utils import from_scipy_sparse_matrix

from tomognn.data import astra_create_sinogram, astra_create_sinogram_w_noise


def build_astra_graph(num_pixels=128, num_detectors=128, num_angles=128, detector_size=1):
    """Build the ASTRA parallel-beam projector and the bipartite detector/pixel graph.

    The ASTRA ('strip') system matrix is used as the weighted adjacency matrix: row i is a
    sinogram bin (LOR), column j is a pixel. Pixel node ids are offset so that they follow
    the sinogram node ids, matching ``tomognn.models.build_nodes_features``.

    Returns (vol_geom, proj_geom, proj_id, edge_index, edge_weight).
    """
    vol_geom  = astra.create_vol_geom(num_pixels, num_pixels)
    proj_geom = astra.create_proj_geom('parallel', detector_size, num_detectors, np.linspace(0,np.pi,num_angles,False))
    proj_id   = astra.create_projector('strip', proj_geom, vol_geom)

    system_matrix_id = astra.projector.matrix(proj_id)
    system_matrix    = astra.matrix.get(system_matrix_id)
    adj_matrix       = sp.csr_matrix(system_matrix)
    # get tensor from adj_matrix:
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
    edge_index = torch.tensor(edge_index.clone().detach())
    edge_index[1] = edge_index[1] + torch.max(edge_index[0]) + 1
    edge_weight = torch.tensor(edge_weight.clone().detach())
    return vol_geom, proj_geom, proj_id, edge_index, edge_weight


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
