#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:26:14 2023

@author: Meysam
"""
import numpy as np
import spacy
import language_tool_python
import nrrd

def pad_texts(texts):
    max_len = 96
    padded_texts = []
    for text in texts:
        words = text.split()
        if len(words) < max_len:
            num_pad = max_len - len(words)
            words += ['#'] * num_pad
        padded_texts.append(' '.join(words))
    return padded_texts

def preprocess_textlist(text_list):
    tool = language_tool_python.LanguageTool('en-US') # load the LanguageTool
    nlp = spacy.load('en_core_web_sm')
    
    # create a dictionary of unique tokens
    # dictionary = {token: idx for idx, token in enumerate(nlp.vocab.strings)}
    # text_list = pad_texts(text_list)
    # sequences_idxs = np.empty((len(text_list), 96))
    decoded_texts = list()
    
    for c, text in enumerate(text_list):
        # lowercase the text
        text = text.lower()
        
        # check for spelling errors and get suggestions
        matches = tool.check(text)
        corrections = [match.replacements[0] for match in matches]
        
        # replace misspelled words with suggested corrections
        for i, match in enumerate(matches):
            start, end = match.offset, match.offset + match.errorLength
            text = text[:start] + corrections[i] + text[end:]
        
        # tokenize and lemmatize the text using Spacy
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc]
        tokens = tokens[:96]
        decoded_text = text = ' '.join(tokens)
        decoded_texts.append(decoded_text)
        
    return decoded_texts



# text_list = ["The quick brown fox jumped over the lazy dog.", "hello, a good chair!"]
# preprocessed_text_list = preprocess_textlist(text_list)


import torch
from transformers import BertModel, BertTokenizer

def embed_textlist(text_list):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    
    # Tokenize input texts
    tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, max_length=64) for text in text_list]
    
    # Pad tokenized texts to maximum length
    max_length = 64
    padded_texts = [text + [0] * (max_length - len(text)) for text in tokenized_texts]
    
    # Convert padded texts to PyTorch tensor
    input_ids = torch.tensor(padded_texts)
    
    # Generate embeddings with lower dimensionality
    with torch.no_grad():
        embeddings = model(input_ids)[0][:, :256]
    
    print(embeddings.shape)  # Output: torch.Size([2, 64, 768])
    return embeddings

def read_gt(nrrd_files_list):
    """
    read a list of nrrd file paths and get them into one tensor!
    """
    data_list = np.empty((len(nrrd_files_list), 4, 32, 32, 32))
    # Loop over each file and load the data
    for c, nrrd_file in enumerate(nrrd_files_list):
        # Load the NRRD file into a numpy array
        # breakpoint()
        data, _ = nrrd.read(nrrd_file)
    
        # Append the data to the list
        data_list[c] = (data)

    # Convert the list to a PyTorch tensor
    data_tensor = torch.tensor(data_list)
    return data_tensor


# embeded_textlist = embed_textlist(preprocessed_text_list)

