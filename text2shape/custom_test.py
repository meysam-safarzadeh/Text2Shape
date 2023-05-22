#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:50:56 2023

@author: Meysam
"""
import torch
from models.test_tokenize import preprocess_textlist, embed_textlist
import numpy as np
import pyvista as pv

def custom_test(model, text_list):
    """
    Inputs: list of text.
    
    Outputs: RGB voxelization of the input text!
    
    """
    device = device = torch.device('cuda')
    
    input_data = preprocess_textlist(text_list)
    input_data = embed_textlist(input_data)
    input_data = torch.tensor(input_data).to(device) # shape: (batch size, 1, 64, 768)
    
    input_data = input_data.unsqueeze(1)

    # Pass the sample through the model and store the output
    model_output = model(input_data)['sigmoid_output']  # Unsqueeze to add a batch dimension
    
    return model_output


def visualize_voxelization(voxel_list, threshold=35):
    """
    Inputs: voxelization tensor with shape (batch_size, 4, 32, 32, 32)
    
    
    """
    
    # Convert tensor to array of uint8
    
    for i in range(len(voxel_list)):
        voxel_input = voxel_list[i].cpu()
        voxels = (voxel_input.detach().numpy() * 255).astype(np.uint8)
    
        # Create a PyVista grid object
        grid = pv.UniformGrid()
        grid.dimensions = voxels.shape[1:]  # Set the dimensions of the grid
        grid.point_data["RGB"] = voxels[:3].transpose(1, 2, 3, 0).reshape(-1, 3)  # Assign the RGB values to the grid's point data
    
        grid = grid.threshold(value=threshold)
        # Visualize the RGB colored voxelization
        grid.plot(scalars="RGB", rgb=True, background="white")
