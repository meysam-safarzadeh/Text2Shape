#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:38:15 2023

@author: Meysam
"""

import torch

def extract_last_output(output, seq_length):
    batch_size = output.shape[0]
    max_length = output.shape[1]
    out_size = output.shape[2]

    index = torch.arange(0, batch_size) * max_length + (seq_length - 1)
    flat = output.view(-1, out_size)
    relevant = flat[index]
    return relevant


def compute_sequence_length(input_batch):
    """Computes sequence length given the input batch.

    Args:
        input_batch: A BxC tensor where B is batch size, C is max caption
            length. 0 indicates the padding, a non-zero positive value indicates a word index.

    Returns:
        seq_length: Tensor of size [batch_size] representing the length of each caption in the current batch.
    """
    # Used represents a BxC tensor where (i, j) element is 1 if the jth word of the ith sample (in the batch) is used/present.
    used = input_batch > 0
    seq_length = torch.sum(used.int(), dim=1)

    return seq_length.int()
