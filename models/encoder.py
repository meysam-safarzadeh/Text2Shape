#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 16:42:31 2023

@author: Meysam
"""
import torch
import torch.nn as nn

class Text_Encoder(nn.Module):
    def __init__(self):
        super(Text_Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=256, out_features=128)


    def forward(self, x):
        # print(x.shape)
        x = x.squeeze()
        
        x = self.conv_layers(x.transpose(1, 2))
        # print(x.shape)
        x, _ = self.gru(x.transpose(1, 2))
        # print(x.shape)
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()
        # print(x.shape, 'here')
        x = self.fc(x)
        # print(x.shape)
        
        regularizer_loss = 0
        for param in self.parameters():
            regularizer_loss += torch.sum(param.pow(2))
        return x + 0.0005 * regularizer_loss

# # To use the model
# model = Text_Encoder()
# input_tensor = torch.randn(32, 1, 64, 768) # assuming a batch size of 32 and input size of (64, 768)
# output_tensor = model(input_tensor) # Transpose the tensor to be in the (batch_size, input_channels, input_length) format expected by the convolution layers

