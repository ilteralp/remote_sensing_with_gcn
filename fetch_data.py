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
NUM_CLASSES = 15
NUM_TRAIN_SAMPLES = 2832
NUM_TEST_SAMPLES = 12197
#TOTAL_SAMPLES = NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES
TOTAL_SAMPLES = 4

"""
Reads given dataset file containing both train and test samples
where first NUM_TRAIN_SAMPLES one belongs to train set. 
"""
def read_dataset():
    with open(NODE_FILE_PATH) as node_file:
        node_data = json.load(node_file)
        # Check total samples
        if len(node_data) != TOTAL_SAMPLES:
            print("Samples length:", len(node_data), "vs", TOTAL_SAMPLES)
            return False
        else:
            """ ====================== x ====================== """
            x = torch.FloatTensor([sample['features'] for sample in node_data])
             
            """ ====================== y ====================== """
            y = torch.IntTensor([sample['label'] for sample in node_data])
            
            """
            # Node ids
            # Read data and turn it into torch tensor
            X = torch.from_numpy(np.asarray([sample['nodeId'] for sample in node_data]))
            print(X)
            print(type(X))
            """
            return True, x, y
        
    
"""
Reads given edges file where relations between nodes are binary 
and two nodes are connected if they belong to the same class. 
"""
def read_edges():
    with open(EDGES_FILE_PATH) as edge_file:
        edge_data = json.load(edge_file)
        if len(edge_data) != NUM_CLASSES:
            print("Classes length", len(edge_data))
            return False
        else:
            edges = []
            edges.append([])
            edges.append([])
            for i, sample_class in enumerate(edge_data):
                class_edges = sample_class[str(i+1)]
                if len(class_edges) > 0:
                    for edge in class_edges:
                        for neighbour in class_edges:
                            if edge != neighbour:
                                edges[0].append(edge)
                                edges[1].append(neighbour)
                #print(sample_class[str(i+1)])
            return True, torch.LongTensor(edges)
   
# Read dataset
ret_val = read_dataset()
if ret_val[0] is True:
    _, x, y = ret_val
    # Read edges
    edge_ret_val = read_edges()
    if edge_ret_val[0] is True:
        edge_index = edge_ret_val[1]
        # print(x)
        # print(y)
        # print(edge_index)
        data = Data(x=x, y=y, edge_index=edge_index)
        print("data:", data)








