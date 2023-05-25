#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:27:30 2023

@author: Meysam
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models.test_tokenize import preprocess_textlist, embed_textlist, read_gt
from models.model import SimpleText2Shape

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load the input data
        # print(self.data.iloc[index, 2])
        input_data = [self.data.iloc[index, 2]]
        input_data = preprocess_textlist(input_data)
        input_data = embed_textlist(input_data)
        input_data = torch.tensor(input_data).to(self.device) # shape: (batch size, 1, 64, 768)

        # Load the ground truth data from file
        gt_file = [self.data.iloc[index, 6]]
        gt_data = read_gt(gt_file).to(self.device) # shape: (batch size, 4, 32, 32, 32)
        gt_data = gt_data.squeeze()/255.0

        return input_data, gt_data