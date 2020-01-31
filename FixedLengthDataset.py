#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:46:25 2020

@author: rog
"""

import torch
from torch_geometric.data import InMemoryDataset

class FixedLengthDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(FixedLengthDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['/home/rog/code/geometric/data/fixed_length.dataset']
    
    def download(self):
        pass
    
    """
    Gather your data into a list of Data objects. Then call self.collate() to 
    compute slices that will be used by the DataLoader object. 
    """
    def process(self):
        
        data_list = []
        
        
