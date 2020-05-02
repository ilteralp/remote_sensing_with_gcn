# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:04:37 2020

@author: melike
"""

import os.path as osp
import numpy as np
import json
import torch

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

ROOT_PATH = './data'
# NUM_TRAIN_SAMPLES = 2832
# NUM_TEST_SAMPLES = 12197
NUM_TRAIN_SAMPLES = 3
NUM_TEST_SAMPLES = 4
TOTAL_SIZE = NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES
TRAIN_NODE_TYPE = 1

""" ======================================== Create Dataset ======================================== """

class TreeInput(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TreeInput, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # The name of the files to find in the self.raw_dir folder in order to skip the download.
    @property
    def raw_file_names(self):
        return [osp.join(self.root, file_name) for file_name in ['edges.txt', 'nodes.txt']] # MC#1
    
    # A list of files in the processed_dir which needs to be found in order to skip the processing.
    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def download(self):
        pass 
    
    """
    Gather data into one Data object for creating only one graph. 
    """
    def process(self):
        edge_path = osp.join(self.root, 'edges.txt')
        node_path = osp.join(self.root, 'nodes.txt')
        
        data = read_tree_input_data(edge_path, node_path)  # MC#2 raw_file_names[0] vs alabilir mi ?
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
        
"""
Returns binary mask whose given indices set to true
and rest are set to false. Taken from io/planetoid.py
"""               
def index_to_mask(index):
    mask = torch.zeros((TOTAL_SIZE, ), dtype=torch.bool)
    mask[index] = 1
    return mask


"""
For a given node, returns its neighbours in COO-format depending node type. 
"""
def get_neighbours(node_id, node_type, neighbours):
    source_nodes = []
    target_nodes = []
    for neigh_id in neighbours:
        source_nodes.append(node_id)
        target_nodes.append(neigh_id)
    return source_nodes, target_nodes
    
"""
Reads given dataset from file into one Data tuple with train and test masks.
"""
def read_tree_input_data(edge_path, node_path):
    
    # Create train and test masks
    train_index = torch.arange(NUM_TRAIN_SAMPLES, dtype=torch.long)
    test_index = torch.arange(NUM_TRAIN_SAMPLES, 
                              NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES,
                              dtype=torch.long)
    
    train_mask = index_to_mask(train_index)
    test_mask = index_to_mask(test_index)
    
    # Read samples
    with open(node_path) as node_file:
        node_data = json.load(node_file)
        with open(edge_path) as edge_file:
            edge_data = json.load(edge_file)
            xs = []                              # node features
            ys = []                              # node labels
            source_edges = []                    # COO format, from-to
            target_edges = []
            for sample in node_data:
                node_id = sample['nodeId']
                xs.append(sample['features'])
                label = sample['label'] - 1      # Labels started from 0 instead of 1
                ys.append(label)
                raw_edges = edge_data[label][str(label+1)]
            
            
    #return train_mask, test_mask
    

    
ROOT_PATH = 'C:\\Users\\melike\\code\\pytorch_geometric\\data\\'
EDGE_PATH = ROOT_PATH + 'edges.txt'
NODE_PATH = ROOT_PATH + 'nodes.txt'
train_mask, test_mask = read_tree_input_data(EDGE_PATH, NODE_PATH)

print('train_mask')
print(train_mask)
print('test_mask')
print(test_mask)
    
    
    
    
    
    
    
    
    
    
    
    
    
    