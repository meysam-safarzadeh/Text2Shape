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
from models.model import SimpleText2Shape
from data.dataset import CustomDataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim


def train(train_loader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    return train_loss

def val(val_loader, model, criterion, optimizer, device):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
    
    val_loss /= len(val_loader.dataset)
    
    return val_loss

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    
    return accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the path to the CSV file
    csv_file = '/home/dulab/Courses/intelligentVisualComputing/Final_project/final_project_CS674/text2shape/data/captions.tablechair.csv'
    
    # Create a dataset from the CSV file
    dataset = CustomDataset(csv_file)
    
    # Set the batch size for the data loader
    batch_size = 32
    
    # Define the ratios for train, validation, and test sets
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    
    # Calculate the number of samples for each set
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Use random_split to split the dataset into train, validation, and test sets
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders for each set
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    # Create an instance of the combined model
    model = SimpleText2Shape()
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and evaluate the model
    for epoch in range(10):
        train_loss = train(train_loader, model, criterion, optimizer, device)
        val_loss = val(val_loader, model, criterion, device)
    
    # Evaluate on the test set
    test_loss = test(model, test_loader, criterion, device)
    print(f'Test loss: {test_loss:.4f}')
















































# # Create a data loader for the dataset
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# index = [2, 10, 5, 6, 7 , 9, 10]
# inputs, targets = dataset.__getitem__(index)

# # Create an instance of the combined model
# model = SimpleText2Shape()
# output_shape = model(inputs.transpose(1, 2))



