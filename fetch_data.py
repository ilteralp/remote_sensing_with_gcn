#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:09:25 2020

@author: rog
"""
import json
import torch
import numpy as np
from torch_geometric.data import Data

ROOT_PATH = "/home/rog/rs/gcn/paths19/test_146/"
NODE_FILE_PATH = ROOT_PATH + "sub_gcn_dataset.txt"
EDGES_FILE_PATH = ROOT_PATH + "sub_edges.txt"
NUM_TRAIN_SAMPLES = 2832
NUM_TRAIN_SAMPLES = 12197

"""
Reads given dataset file containing both train and test samples
where first NUM_TRAIN_SAMPLES one belongs to train set. 
"""
def read_dataset():
    with open(NODE_FILE_PATH) as node_file:
        node_data = json.load(node_file)
        # Check total samples
        
        """
        # X
        for sample in node_data:
            node_feat = np.asarray(sample['features'])
            print(node_feat)
            print(type(node_feat))
        """
        # y
        #y = torch.from_numpy(np.asarray([sample['label'] for sample in node_data]))
        y = torch.IntTensor([sample['label'] for sample in node_data])
        print(y)
        print(y.type(), type(y[0]), y.data.tolist())
        """
        # Node ids
        # Read data and turn it into torch tensor
        X = torch.from_numpy(np.asarray([sample['nodeId'] for sample in node_data]))
        print(X)
        print(type(X))
        """
        
    
"""
Reads given edges file where nodes between relations are binary 
and two nodes are connected if they belong to the same class. 
"""
def read_edges():
    with open(EDGES_FILE_PATH) as edge_file:
        edge_data = json.load(edge_file)
        
read_dataset()