#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:46:25 2020

@author: rog
"""

import json
import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.preprocessing import LabelEncoder
import pandas as pd

NUM_CLASSES = 15
NUM_TRAIN_SAMPLES = 2832
NUM_TEST_SAMPLES = 12197
#TOTAL_SAMPLES = NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES
TOTAL_SAMPLES = 15029

"""
See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/geometry.html
"""
class GraphInputDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphInputDataset, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    # The name of the files to find in the self.raw_dir folder in order to skip the download.
    @property
    def raw_file_names(self):
        return ['raw.txt']
    
    # A list of files in the processed_dir which needs to be found in order to skip the processing.
    @property
    def processed_file_names(self):
        return ['processed.txt']
    
    def download(self):
        pass
    
    """
    Gather your data into a list of Data objects. Then call self.collate() to 
    compute slices that will be used by the DataLoader object. 
    """
    def process(self):
        # Read data into huge `Data` list.
        is_data_fetched, data_list = read_dataset()
        if is_data_fetched:
            print("data_list len:", len(data_list))
            """
            for data in data_list:
                print("x:", data.x, "y:", data.y, "edge_index:", data.edge_index)
            """
            
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
    
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            

"""
Reads given dataset file which includes all data samples (contains test and 
train set together) and edge file which contains relations between nodes 
where two nodes are connected if they belong to the same class. Returns 
resulting graph that is undirected. 
"""
def read_dataset():
    with open(NODE_FILE_PATH) as node_file: # Read node file. 
        node_data = json.load(node_file)
        # Check total number samples
        if len(node_data) != TOTAL_SAMPLES:
            print("Samples length:", len(node_data), "vs", TOTAL_SAMPLES)
            return False, []
        else:
            with open(EDGES_FILE_PATH) as edge_file: # Read edge file. 
                edge_data = json.load(edge_file)
                # Check number of classes
                if len(edge_data) != NUM_CLASSES:
                    #print("Classes length", len(edge_data))
                    return False, []
                else:
                    data_list = []
                    # Fetch each sample seperately
                    for sample in node_data:
                        node_id = sample['nodeId']
                        features = sample['features']
                        label = sample['label']
                        raw_edges = edge_data[label-1][str(label)] # Edges include id of node being processed, some remove it. 
                        neighbours = [neighbour_id for neighbour_id in raw_edges if neighbour_id != node_id]
                        data = Data(x=features, y=label, edge_index=neighbours)
                        #print("node_id:", node_id, "x:", features, "y:", label, "edge_index:", neighbours)
                        data_list.append(data)
                    return True, data_list 
                
        
""" ======================= Read dataset ======================= """
# Constants
test_id = 146   # Variable length input
print("Test", test_id)
ROOT_PATH = "/home/rog/rs/gcn/paths19/test_" + str(test_id) + "/"
NODE_FILE_PATH = ROOT_PATH + "gcn_dataset.txt"
EDGES_FILE_PATH = ROOT_PATH + "edges.txt"

"""
info_path = ROOT_PATH +"info.txt"
"""
dataset = GraphInputDataset(ROOT_PATH)
# Create train & test datasets 
train_dataset = dataset[:NUM_TRAIN_SAMPLES]
test_dataset = dataset[NUM_TRAIN_SAMPLES:]
train_dataset = train_dataset.shuffle()
print(len(train_dataset), len(test_dataset))









