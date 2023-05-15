#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 16:45:06 2023

@author: Meysam
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class Text2ShapeGenerator(nn.Module):

    def __init__(self, is_training=False, name='t2s_generator_1'):
        super(Text2ShapeGenerator, self).__init__()
        self.is_training = is_training

        # Conv1
        self.fc1 = nn.Linear(128, 512 * 4 * 4 * 4)
        self.fc1_bn = nn.BatchNorm1d(512 * 4 * 4 * 4)

        # Conv2
        self.conv_transpose2 = nn.ConvTranspose3d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv_transpose2_bn = nn.BatchNorm3d(512)

        # Conv3
        self.conv_transpose3 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv_transpose3_bn = nn.BatchNorm3d(256)

        # Conv4
        self.conv_transpose4 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_transpose4_bn = nn.BatchNorm3d(128)

        # Conv5
        self.conv_transpose5 = nn.ConvTranspose3d(128, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        x = inputs
        # print('\t\tinput', x.shape)

        # Conv1
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        # print(x.shape)
        x = x.view(-1, 512, 4, 4, 4)
        # print(x.shape)

        # Conv2
        x = self.conv_transpose2(x)
        x = self.conv_transpose2_bn(x)
        x = F.relu(x)

        # Conv3
        x = self.conv_transpose3(x)
        x = self.conv_transpose3_bn(x)
        x = F.relu(x)

        # Conv4
        x = self.conv_transpose4(x)
        x = self.conv_transpose4_bn(x)
        x = F.relu(x)
        # print(x.shape)

        # Conv5
        logits = self.conv_transpose5(x)
        sigmoid_output = torch.sigmoid(logits)

        return {'sigmoid_output': sigmoid_output, 'logits': logits}




# Create an instance of the network
net = Text2ShapeGenerator()

# Define a random input tensor with the appropriate shape
input_shape = (2, 128)
text_encoding_with_noise = torch.rand(input_shape)

# Pass the input tensor through the network
output_dict = net(text_encoding_with_noise)

# Print the shape of the output tensor
print('Sigmoid output shape:', output_dict['sigmoid_output'].shape)
print('Logits shape:', output_dict['logits'].shape)
