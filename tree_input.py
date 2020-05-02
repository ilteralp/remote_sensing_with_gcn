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
NUM_TRAIN_SAMPLES = 4
NUM_TEST_SAMPLES = 3
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
        class_neigh_path = osp.join(self.root, 'class_neighbours.txt')
        level_neigh_path = osp.join(self.root, 'level_neighbours.txt')
        node_path = osp.join(self.root, 'feats.txt')

        ret_val, data = read_tree_input_data(class_neigh_path, level_neigh_path, node_path)  # MC#2 raw_file_names[0] vs alabilir mi ?
        if ret_val:
            data = data if self.pre_transform is None else self.pre_transform(data)
            torch.save(self.collate([data]), self.processed_paths[0])
        else:
            print("Could not read dataset")
            return
        
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
def get_neighbours(node_id, neighbours, source_nodes, target_nodes):
    for neigh_id in neighbours:
        if neigh_id != node_id:
            source_nodes.append(node_id)
            target_nodes.append(neigh_id)
    return source_nodes, target_nodes
    
"""
Reads given dataset from file into one Data tuple with train and test masks.
"""
def read_tree_input_data(class_neigh_path, level_neigh_path, node_path):
    
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
        if len(node_data) != TOTAL_SIZE:
            print("Expected", TOTAL_SIZE, "samples, given", len(node_data), "samples!")
            return False, None
        with open(class_neigh_path) as class_neigh_file:
            class_neigh_data = json.load(class_neigh_file)
            with open(level_neigh_path) as level_neigh_file:
                level_neigh_data = json.load(level_neigh_file)
                xs = []                                         # node features
                ys = []                                         # node labels
                from_nodes = []                                 # COO format, from-to
                to_nodes = []
                for sample in node_data:
                    node_id = sample['nodeId']
                    xs.append(sample['features'])
                    label = sample['label'] - 1                 # labels started from 0 instead of 1
                    ys.append(label)
                    level = sample['level']
                    level_node_ids = level_neigh_data[level][str(level)]
                    from_nodes, to_nodes = get_neighbours(node_id, level_node_ids, from_nodes, to_nodes)
                    node_type = sample['train']
                    if node_type == TRAIN_NODE_TYPE:
                        class_node_ids = class_neigh_data[label][str(label+1)]
                        from_nodes, to_nodes = get_neighbours(node_id, class_node_ids, from_nodes, to_nodes)
                        
                x = torch.from_numpy(np.array(xs)).to(torch.float)
                y = torch.from_numpy(np.array(ys)).to(torch.long) 
                edge_index = torch.from_numpy(np.array([from_nodes, to_nodes])).to(torch.long)
                print(edge_index)
                data = Data(x=x, y=y, edge_index=edge_index)
                data.train_mask = train_mask
                data.test_mask = test_mask
                return True, data
    return False, None

# Create dataset
dataset = TreeInput(ROOT_PATH)
# Check there is only one graph
assert len(dataset) == 1
    
# Create network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
    
    
    
    
    