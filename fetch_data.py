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

#ROOT_PATH = "/home/rog/rs/gcn/paths19/test_146/"
ROOT_PATH = "./data/"
NODE_FILE_PATH = ROOT_PATH + "sub_gcn_dataset.txt"
EDGES_FILE_PATH = ROOT_PATH + "sub_edges.txt"
NUM_CLASSES = 15
NUM_TRAIN_SAMPLES = 2832
NUM_TEST_SAMPLES = 12197
#TOTAL_SAMPLES = NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES
TOTAL_SAMPLES = 6

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
                    print("Classes length", len(edge_data))
                    return False, []
                else:
                    data_list = []
                    # Fetch each sample seperately
                    for sample in node_data:
                        node_id = sample['nodeId']
                        features = sample['features']
                        label = sample['label']
                        raw_edges = edge_data[label-1][str(label)] # Edges include id of node being processed, some remove it. 
                        neighbours = [neigbour_id for neigbour_id in raw_edges if neigbour_id != node_id]
                        data = Data(x=features, y=label, edge_index=neighbours)
                        print("node_id:", node_id, "x:", features, "y:", label, "edge_index:", neighbours)
                        data_list.append(data)
                    return True, data_list
                    
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
is_data_fetched, data_list = read_dataset()
if is_data_fetched:
    print("data_list len:", len(data_list))
    for data in data_list:
        print("x:", data.x, "y:", data.y, "edge_index:", data.edge_index)

"""
if ret_val[0] is True:
    _, x, y = ret_val
    # Read edges
    edge_ret_val = read_edges()
    if edge_ret_val[0] is True:
        edge_index = edge_ret_val[1]
        # print(x)
        print(y)
        print(edge_index)
        data = Data(x=x, y=y, edge_index=edge_index)
        print("data:", data)
"""








