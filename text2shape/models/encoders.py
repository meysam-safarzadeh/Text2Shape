#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 16:42:31 2023

@author: Meysam
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_utils import extract_last_output, compute_sequence_length

class CNNRNNTextEncoder(nn.Module):
    """CNN-RNN Text Encoder network.
    """

    def __init__(self, is_training, reuse=False, name='text_encoder_example'):
        super(CNNRNNTextEncoder, self).__init__(is_training, reuse=reuse, name=name)
        self._embedding_size = 128
        self._margin = 1

        # Define layers for the convolutional network
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
                                          nn.ReLU(),
                                          nn.BatchNorm2d(128),
                                          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
                                          nn.ReLU(),
                                          nn.BatchNorm2d(256)])

        # Define the GRU cell for the RNN network
        self.rnn_cell = nn.GRU(input_size=256, hidden_size=256)

        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)

    def forward(self, inputs_dict):
        """Builds the RNN text encoder.

        Returns:
            encoder_output: The output of the text encoder.
        """

        caption_batch = inputs_dict['caption_batch']
        embedding = inputs_dict['embedding_batch']
        seq_length = compute_sequence_length(caption_batch)

        # Pass the embedding through the convolutional layers
        net = embedding.unsqueeze(1)  # Add a channel dimension
        for layer in self.conv_layers:
            net = layer(net)
            if isinstance(layer, nn.ReLU):
                net = F.dropout(net, p=0.5, training=self.is_training)  # Add dropout after ReLU
            if isinstance(layer, nn.BatchNorm2d):
                net = F.batch_norm(net, training=self.is_training)

        net = F.max_pool2d(net, kernel_size=(seq_length, 1))  # Apply max pooling over the sequence length dimension

        # Pass the resulting feature map through the RNN network
        outputs, final_state = self.rnn_cell(net.squeeze(2).permute(1, 0, 2), hx=None)  # Reshape for RNN input

        # Extract the final output of the RNN network
        net = extract_last_output(outputs.permute(1, 0, 2), seq_length)

        # Pass the final output through the fully connected layers
        net = self.fc1(net)
        net = F.relu(net)
        net = self.fc2(net)

        return {'encoder_output': net}

    @property
    def embedding_size(self):
        return self._embedding_size
