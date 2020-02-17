#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:46:25 2020

@author: rog
"""

import torch
from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd

"""
See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/geometry.html
"""
class GraphInputDataset(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None):
        super(GraphInputDataset, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    # The name of the files to find in the self.raw_dir folder in order to skip the download.
    @property
    def raw_file_names(self):
        return ['train.txt', 'test.txt']
    
    # A list of files in the processed_dir which needs to be found in order to skip the processing.
    @property
    def processed_file_names(self):
        #return ['data.pt']
        return ['train.pt', 'test.pt']
    
    def download(self):
        pass
    
    """
    Gather your data into a list of Data objects. Then call self.collate() to 
    compute slices that will be used by the DataLoader object. 
    """
    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) # Add train or test
        
""" ======================= Read dataset ======================= """
# Constants
test_id = 146   # Variable length input
print("Test", test_id)
ROOT_PATH = "/home/rog/rs/gcn/paths19/test_" + str(test_id) + "/"
"""
X_train_feat_path = ROOT_PATH + "train.txt"
X_test_feat_path = ROOT_PATH + "test.txt"
info_path = ROOT_PATH +"info.txt"

df_tr = pd.read_csv(X_train_feat_path, header=None)
df_test = pd.read_csv(X_test_feat_path, header=None)
"""
train_dataset = GraphInputDataset("/home/rog/rs/gcn/paths19/test_143/", train=True)
#print(train_dataset.processed_dir)
#test_dataset = GraphInputDataset(ROOT_PATH, train=False)









