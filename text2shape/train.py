#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:43:02 2023

@author: Meysam
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models.test_tokenize import preprocess_textlist, embed_textlist, read_gt


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load the input data
        input_data = self.data.iloc[index, 2].tolist()
        # breakpoint()
        input_data = preprocess_textlist(input_data)
        input_data = embed_textlist(input_data)
        input_data = torch.tensor(input_data) # shape: (batch size, 128)

        # Load the ground truth data from file
        gt_file = self.data.iloc[index, 6].tolist()
        gt_data = read_gt(gt_file) # shape: (batch size, 4, 32, 32, 32)

        return input_data, gt_data

# Set the path to the CSV file
csv_file = '/home/dulab/Courses/intelligentVisualComputing/Final_project/final_project_CS674/text2shape/data/captions.tablechair.csv'

# Create a dataset from the CSV file
dataset = CustomDataset(csv_file)

# Set the batch size for the data loader
batch_size = 32

# Create a data loader for the dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

index = [2, 10, 5, 6, 7 , 9, 10]
inputs, targets = dataset.__getitem__(index)

