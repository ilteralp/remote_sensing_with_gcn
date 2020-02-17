#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:39:56 2020

@author: rog
"""
import torch
from torch_geometric.data import InMemoryDataset, Data

class SampleDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SampleDataset, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)
        
    # The name of the files to find in the self.raw_dir folder in order to skip the download.
    @property
    def raw_file_names(self):
        return ['sample_raw.txt']
    
        # A list of files in the processed_dir which needs to be found in order to skip the processing.
    @property
    def processed_file_names(self):
        return ['sample_processed.txt']
    
    def download(self):
        pass
    
    def process(self):
        is_data_fetched, data_list = read_sample_dataset()
        if is_data_fetched:
            print("data_list len:", len(data_list))
            for data in data_list:
                print("x:", data.x, "y:", data.y, "edge_index:", data.edge_index)
                
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
    
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
                
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0]) 

def read_sample_dataset():
    feats = [[2,1], [5,6], [3, 7], [12, 0]]
    labels = [0, 1, 0, 1]
    edges = [[0, 0, 1, 2, 3], [1, 3, 0, 1, 2]]
    
    data_list = []
    for node_index, feat in enumerate(feats):
        x =  torch.FloatTensor(feat)
        y = torch.FloatTensor(labels[node_index]) # Float olmalı !
        e_from = []
        e_to = []
        for index, node_id in enumerate(edges[0]):
            if node_id == node_index:
                e_from.append(node_id)
                e_to.append(edges[1][index])
        edge_index = torch.LongTensor([e_from, e_to])
        data_list.append(Data(x=x, y=y, edge_index=edge_index))
    return True, data_list
    
ROOT_PATH = "/home/rog/code/geometric/tmp"
dataset = SampleDataset(ROOT_PATH)
NUM_TRAIN_SAMPLES = 2
train_dataset = dataset[:NUM_TRAIN_SAMPLES]
test_dataset = dataset[NUM_TRAIN_SAMPLES:]
print("num_features:", train_dataset.num_features, test_dataset.num_features)










