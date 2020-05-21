# -*- coding: utf-8 -*-
"""
Created on Thu May 21 01:51:30 2020

@author: melike
"""

import os.path as osp
import numpy as np
import json
import torch

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import GCNConv
from torch_sparse import coalesce
import torch.nn.functional as F

TRAIN_NODE = 1
TEST_NODE = 2
ROOT_PATH = "C:\\Users\\melike\\RS\\gcn\\paths19\\test_149"
NODE_FILE_NAME = '68k_pixel_as_node.txt'
#ROOT_PATH = './data'

class PixelNode(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PixelNode, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    # The name of the files to find in the self.raw_dir folder in order to skip the download.
    @property
    def raw_file_names(self):
        return osp.join(self.root, NODE_FILE_NAME)
    
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
        node_file_path = osp.join(self.root, NODE_FILE_NAME)
        ret_val, data = read_pixel_node_data(node_file_path)
        if ret_val:
            print(data)
            data = data if self.pre_transform is None else self.pre_transform(data)
            torch.save(self.collate([data]), self.processed_paths[0])
        else:
            print("Could not read dataset")
            return
        
"""
Reads pixel-nodes from its file where 
some of the nodes are not labeled. 
"""
def read_pixel_node_data(node_file_path):
    with open(node_file_path) as node_file:
        node_data = json.load(node_file)
        xs = []                              # node features
        ys = []
        node_ids = []
        from_nodes = []                      # COO format, from-to relation
        to_nodes = []
        edge_weights = []
        train_mask = torch.zeros((len(node_data), ), dtype=torch.bool)
        test_mask = torch.zeros((len(node_data), ), dtype=torch.bool)
        for sample in node_data:
            xs.append(sample['fs'])
            ys.append(sample['label'])   # labels [1,15], start them from 0.
            node_id = sample['id']
            node_ids.append(node_id)
            node_type = sample['type']
            if node_type == TRAIN_NODE:
                train_mask[node_id] = 1
            elif node_type == TEST_NODE:
                test_mask[node_id] = 1
            ns = sample['ns']
            for neigh in ns:
                from_nodes.append(node_id)
                to_nodes.append(neigh[0])
                edge_weights.append(neigh[1])
                
        x = torch.from_numpy(np.array(xs)).to(torch.float)
        y = torch.from_numpy(np.array(ys)).to(torch.long)
        edge_index = torch.from_numpy(np.array([from_nodes, to_nodes])).to(torch.long)
        # edge_attr = torch.from_numpy(np.array(edge_weights)).to(torch.float)

        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = train_mask
        data.test_mask = test_mask
        return True, data
    return False, None
        
# FILE_PATH = osp.join(ROOT_PATH, 'pixel_as_node.txt')
# ret_val, data = read_pixel_node_data(FILE_PATH)
# if ret_val:
#     print(data)
#     for key, val in data:
#         print(key)
#         print(val)
# else:
#     print('could not read!')

dataset = PixelNode(ROOT_PATH)
data = dataset[0]
print('num_nodes', data.num_nodes, 'num_classes', dataset.num_classes, 'dataset_len', len(dataset))
print('contains_self_loops', data.contains_self_loops())
print('contains_isolated_nodes', data.contains_isolated_nodes())

# Check there is only one graph
assert len(dataset) == 1

# Create network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        
    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
device = torch.device('cpu')
data = dataset[0]
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
    
for epoch in range(1, 25):
    train()
    train_acc, test_acc = test()
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, test_acc))      
        
        
        
        
        
        
        
        
        
        
        
    