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
import torchvision.models as models
from models.model import SimpleText2Shape

def custom_test(model, text_list):
    """
    Inputs: list of text.
    
    Outputs: RGB voxelization of the input text!
    
    """
    device = device = torch.device('cuda')
    
    input_data = preprocess_textlist(text_list)
    input_data = embed_textlist(input_data)
    input_data = torch.tensor(input_data).to(device) # shape: (batch size, 1, 64, 768)
    # Unsqueeze to add a batch dimension
    input_data = input_data.unsqueeze(1)

    # Pass the sample through the model and store the output
    output_dict = model(input_data)
    sigmoid_output = output_dict['sigmoid_output']
    
    return sigmoid_output


def visualize_voxelization(voxel_list, text_list, threshold=35):
    """
    Inputs: voxelization tensor with shape (batch_size, 4, 32, 32, 32)
    
    
    """
    
    # Convert tensor to array of uint8
    
    for i in range(len(voxel_list)):
        voxel_input = voxel_list[i].cpu()
        voxels = (voxel_input.detach().numpy() * 255.0).astype(np.uint8)
    
        # Create a PyVista grid object
        grid = pv.UniformGrid()
        grid.dimensions = voxels.shape[1:]  # Set the dimensions of the grid
        grid.point_data["RGB"] = voxels[:3].transpose(1, 2, 3, 0).reshape(-1, 3)  # Assign the RGB values to the grid's point data
        grid = grid.threshold(value=threshold)
        
        # Visualize the RGB colored voxelization
        plotter = pv.Plotter(off_screen=False)
        plotter.add_mesh(grid)
        
        # Add different lights to the plotter
        light = pv.Light(color='cyan', light_type='headlight')
        plotter.add_light(light)
        plotter.add_light(pv.Light(color='#FF00AA')) 
        
        # Change the camera position
        # camera_position = [(100, 100, 100), (0, 0, 0), (0, 1, 0)]  # Define the new camera position
        # plotter.camera_position = camera_position
        
        # Display the plot
        print(text_list[i])
        plotter.show()
        
        



# Load the model
model_path = 'checkpoints/checkpoint_20.pth.tar'
checkpoint = torch.load(model_path, map_location=torch.device('cuda'))
model = SimpleText2Shape().to('cuda')

# Load the model weights from the checkpoint
model.load_state_dict(checkpoint['state_dict'])


# test on your own inputs
text_list = ['big table desk', 
             'A gray colored chair that has a curved back seat with',
             'Circular table, I would expect to see couches surrounding this type of table.',
             'super large table',
             'A brown color rectangular wooden table with four design',
             'tall big, and cozy office table',
             'big chair with arm rest', 
             'big chair with very very super long legs',
             'seat without back support',
             'seat with back support ',
             'simple sofa with 4 legs',
             'simple sofa with 5 leg']
output_voxels = custom_test(model, text_list)
visualize_voxelization(output_voxels, text_list, threshold=9)

