# -TomoGNN- Linking the Dots: Pixel-Detectors Associations for Improved PET Direct Image Reconstruction.

## Overview

TomoGNN introduces a state-of-the-art approach for Positron Emission Tomography (PET) image reconstruction using Graph Neural Networks (GNNs). It addresses limitations of traditional and deep learning-based methods by providing high-fidelity reconstructions with minimal noise even with sparse data.

## Introduction to PET Imaging Challenges

PET imaging faces challenges due to its inherently noisy and ill-posed nature. Conventional model-based iterative reconstruction (MBIR) methods are effective but computationally intensive. Deep learning approaches often suffer from over-smoothing and lack generalizability.

## TomoGNN: Bridging the Gap

TomoGNN formulates the PET inverse problem within a graph-theoretical framework, modeling the relationship between lines of response (LORs) and pixels using GNNs. This leads to accurate and efficient reconstructions that preserve anatomical structures without over-smoothing.

## Problem Formulation

The PET reconstruction problem is approached through a Poisson noise model, emphasizing the need for accurately inferring the underlying activity distribution from measured projection data (sinogram), considering system resolution, attenuation, and random and scatter events.

## Architecture and Training

TomoGNN's architecture consists of three main components:
1. **Sinogram-to-Sinogram Denoising (CNN):** Pre-processes the sinogram to reduce noise and improve data quality.
2. **Sinogram-to-Image Mapping (GCN):** A novel single-layer GCN effectively projects sinogram data into the image domain.
3. **Image-to-Image Refinement (CNN):** Further refines the mapped image to enhance local features and details.

![TomoGNN Architecture](./Tomo.eps)


The model was trained on a dataset comprising PET images with artificially introduced lesions, using Mean Squared Error (MSE) and Gradient Difference Loss (GDL) to preserve high-frequency image features.

## Results and Benefits

- **Efficient and Accurate Reconstruction:** TomoGNN outperforms traditional MBIR and other deep learning methods, providing detailed images with controlled noise.
- **Minimal Model Complexity:** Despite superior performance, TomoGNN requires fewer trainable parameters, making it efficient in memory and computational resources.
- **Robustness to Varied Data Conditions:** Capable of handling sinograms of any dimension and adaptable to different noise levels and counts, demonstrating exceptional generalization capabilities.

## Usage and Training

TomoGNN was trained using the Adam optimizer over 300 epochs, with specific attention to data augmentation techniques for model robustness. Training utilized a sophisticated loss function balancing reconstruction fidelity with the preservation of image textures and edges.
