#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:46:29 2023

@author: Meysam
"""

from models.generator import Text2ShapeGenerator
from models.encoder import Text_Encoder
import torch
import torch.nn as nn

class SimpleText2Shape(nn.Module):
    def __init__(self):
        super(SimpleText2Shape, self).__init__()
        self.text_encoder = Text_Encoder()
        self.text2shape_generator = Text2ShapeGenerator()
        
        # Move the model to CUDA if available
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()
            self.text2shape_generator = self.text2shape_generator.cuda()

    def forward(self, input_text):
        encoded_text = self.text_encoder(input_text)
        generated_shape = self.text2shape_generator(encoded_text)
        return generated_shape




