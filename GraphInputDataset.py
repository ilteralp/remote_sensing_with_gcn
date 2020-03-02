#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:46:25 2020

@author: rog
"""

import json
import torch
from torch_geometric.data import InMemoryDataset, Data, Batch, NeighborSampler
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import os
import os.path as osp
import numpy as np

NUM_CLASSES = 15
NUM_TRAIN_SAMPLES = 2832
NUM_TEST_SAMPLES = 12197
TOTAL_SAMPLES = NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES
TOTAL_SAMPLES = 4
# if TOTAL_SAMPLES == 3:
#     NUM_TRAIN_SAMPLES = 3
    
# Constants
test_id = 146   # Variable length input
print("Test", test_id)
ROOT_PATH = "/home/rog/rs/gcn/paths19/test_" + str(test_id) + "/"

""" ======================================== Create Dataset ======================================== """

"""
See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/geometry.html
"""
class GraphInputDataset(InMemoryDataset):
    def __init__(self, root, train, transform=None, pre_transform=None):
        
        super(GraphInputDataset, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        
        # print("*" * 50)
        # print("self.data")
        # for key, val in self.data:
        #     print(key, ":", val)
        #     print(key, "shape:", val.shape)
        # print("*" * 50)
        # print("num_nodes:", self.data.num_nodes)
        # print("num_node_features:", self.data.num_node_features)

    # The name of the files to find in the self.raw_dir folder in order to skip the download.
    @property
    def raw_file_names(self):
        return ['{}_edges.txt'.format(file_name) for file_name in ['train', 'test']]
    
    # A list of files in the processed_dir which needs to be found in order to skip the processing.
    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']
    
    def download(self):
        pass 
    
    """
    Gather your data into a list of Data objects. Then call self.collate() to 
    compute slices that will be used by the DataLoader object. 
    """
    def process(self):
        print("Paths:")
        for split, processed_path in zip (['train', 'test'], self.processed_paths):
            node_file_path = osp.join(self.root, '{}_gcn_dataset.txt'.format(split))
            edge_file_path = osp.join(self.root, '{}_edges.txt'.format(split))
            print(node_file_path, edge_file_path)
            # Read data into huge `Data` list.
            is_data_fetched, data_list = read_dataset(node_file_path, edge_file_path)
            if is_data_fetched:
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
        
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]
                
                data, slices = self.collate(data_list)
                torch.save((data, slices), processed_path)
            else:
                print("Dataset cannot be read! Given", node_file_path, "and", edge_file_path)
            

"""
Reads given dataset file which includes all data samples (contains test and 
train set together) and edge file which contains relations between nodes 
where two nodes are connected if they belong to the same class. Returns 
resulting graph that is undirected. 
"""
def read_dataset(node_file_path, edge_file_path):
    with open(node_file_path) as node_file: # Read node file. 
        node_data = json.load(node_file)
        with open(edge_file_path) as edge_file: # Read edge file. 
            edge_data = json.load(edge_file)
            # Check number of classes
            if len(edge_data) != NUM_CLASSES:
                print("Classes length", len(edge_data))
            else:
                data_list = []
                # Fetch each sample seperately
                for sample in node_data:
                    node_id = sample['nodeId']
                    features = torch.FloatTensor(sample['features']) # 1D, convert to 2D
                    features = features.view(1, features.shape[0])   # Data expects [num_nodes, num_node_features]
                    label = sample['label'] - 1                      # -1 since classes start from 1
                    raw_edges = edge_data[label][str(label+1)]       # Edges include id of node being processed, so remove it. 
                    e_from = []
                    e_to = []
                    for neighbour_id in raw_edges:
                        if neighbour_id != node_id:
                            e_from.append(node_id)
                            e_to.append(neighbour_id)
                    edge_index = torch.LongTensor([e_from, e_to])
                    data = Data(x=features, y=torch.LongTensor([label]), edge_index=edge_index)
                    print("shapes:", data.x.size(), data.y.size(), data.edge_index.size())
                    data_list.append(data)
                return True, data_list
        return False, []
    
"""
Reads given dataset from file into one tuple
"""
def read_dataset_one_tuple(node_file_path, edge_file_path):
    with open(node_file_path) as node_file: # Read node file. 
        node_data = json.load(node_file)
        with open(edge_file_path) as edge_file: # Read edge file. 
            edge_data = json.load(edge_file)
            # Check number of classes
            if len(edge_data) != NUM_CLASSES:
                print("Classes length", len(edge_data))
            # convert numpy arrays to tensor
            else:
                xs = []
                ys = []
                source_edges = []
                target_edges = []
                for sample in node_data:
                    node_id = sample['nodeId']
                    xs.append(sample['features'])
                    label = sample['label'] - 1
                    ys.append(label)
                    raw_edges = edge_data[label][str(label+1)]
                    e_from = []
                    e_to = []
                    for neighbour_id in raw_edges:
                        if neighbour_id != node_id:
                            source_edges.append(node_id)
                            target_edges.append(neighbour_id)
                for name, item in zip(['xs', 'ys', 'edge_indices'], [xs, ys, [source_edges, target_edges]]):
                    print(name)
                    print(item)
                print("*" * 50)
                x = torch.from_numpy(np.array(xs)).to(torch.float)
                y = torch.from_numpy(np.array(ys)).to(torch.long)
                edge_indices = torch.from_numpy(np.array([source_edges, target_edges])).to(torch.long)
                return True, Data(x=x, y=y, edge_indices=edge_indices)
    return False, None

NODE_FILE_PATH = ROOT_PATH + 'sample_gcn_dataset.txt'
EDGE_FILE_PATH = ROOT_PATH + 'sample_edges.txt'
ret_val, data = read_dataset_one_tuple(NODE_FILE_PATH, EDGE_FILE_PATH)

# Create dataset
train_dataset = GraphInputDataset(ROOT_PATH, train=True)
test_dataset = GraphInputDataset(ROOT_PATH, train=False)


for dataset, name in zip ([train_dataset, test_dataset], ['train_dataset', 'test_dataset']):
    print("+" * 60)
    print(name)
    print('dataset.data type', type(dataset.data), 'content', dataset.data)
    print("size:", dataset.data.x.size(), "x.size(0):", dataset.data.x.size(0))
    print("num_classes:", dataset.num_classes)
    print("num_features:", dataset.data.num_features)
    print("contains_isolated_nodes:", dataset.data.contains_isolated_nodes())
    print("contains_self_loops:", dataset.data.contains_self_loops())
    print("is_coalesced:", dataset.data.is_coalesced())
    print("is_undirected:", dataset.data.is_undirected())
    print("+" * 60)

# Print content
for dataset, name in zip ([train_dataset, test_dataset], ['train_dataset', 'test_dataset']):
    print("@" * 50)
    print(name)
    for key, val in dataset.data:
        print(key, val)
    print("@" * 50)
    


# Check number of classes & features of nodes in datasets 
assert train_dataset.num_classes == test_dataset.num_classes
assert train_dataset.num_features == test_dataset.num_features


""" ======================================== Build a GNN ======================================== """
# from torch_geometric.nn import SplineConv

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = SplineConv(dataset.num_features, 16, dim=1, kernel_size=2)
#         self.conv2 = SplineConv(16, dataset.num_classes, dim=1, kernel_size=2)

#     def forward(self):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         x = F.dropout(x, training=self.training)
#         x = F.elu(self.conv1(x, edge_index, edge_attr))
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index, edge_attr)
#         return F.log_softmax(x, dim=1)
    
    

""" ======================================== Build a GNN ======================================== """
# See https://github.com/rusty1s/pytorch_geometric/blob/master/examples/enzymes_topk_pool.py

class Net(torch.nn.Module):
    def __init__(self):
        print("Network")
        print("num_features:", train_dataset.num_features)
        print("num_classes:", train_dataset.num_classes)
        super(Net, self).__init__()
        self.conv1 = GraphConv(train_dataset.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, train_dataset.num_classes)
        
    def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            x = x1 + x2 + x3
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.lin2(x))
            x = F.log_softmax(self.lin3(x), dim=-1)
    
            return x
        
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("device:", device)
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        print('output.size', output.size(), 'target.size', data.y.size())
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

train_loader = DataLoader(train_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)


for loader, name in zip([train_loader, test_loader], ['train_loader', 'test_loader']):
    print("+" * 100)
    print(name)
    for num, sample in enumerate(loader):
        print("sample:", num)
        for key, val in sample:
            print('\t', key, val)
    print("+" * 100)

for epoch in range(1, 201):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, test_acc))







