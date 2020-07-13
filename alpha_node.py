# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:08:20 2020

@author: melike
"""

import os.path as osp
import numpy as np
import random
import os
import json
import torch
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import GCNConv
from torch_sparse import coalesce
import torch.nn.functional as F
import Constants

class AlphaNode(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(AlphaNode, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    # The name of the files to find in the self.raw_dir folder in order to skip the download.
    @property
    def raw_file_names(self):
        return osp.join(self.root, Constants.NODE_FILE_NAME)
    
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
        node_file_path = osp.join(self.root, Constants.ALPHA_NODE_FEATS_PATH)
        edge_file_path = osp.join(self.root, Constants.ALPHA_EDGES_PATH)
        
        ret_val, data, map_ids = read_alpha_node_data(node_file_path, edge_file_path)
        if ret_val:
            print(data)
            data = data if self.pre_transform is None else self.pre_transform(data)
            torch.save(self.collate([data]), self.processed_paths[0])
        else:
            print("Could not read dataset")
            return
        
def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def read_alpha_node_data(node_file_path, edge_file_path):
    with open(node_file_path) as node_file:
        with open(edge_file_path) as edge_file:
            node_data = json.load(node_file)
            edge_data = json.load(edge_file)
            xs = []                              # node features
            ys = []
            node_ids = []
            from_nodes = []                      # COO format, from-to relation
            to_nodes = []
            #edge_weights = []
            
            len_data = len(node_data)
            # train_end = int(len(node_data) * Constants.ALPHA_TRAIN_PERCENT)
            # train_index = torch.arange(train_end, dtype=torch.long)
            # test_index = torch.arange(train_end, len(node_data), dtype=torch.long)
            # train_mask = index_to_mask(train_index, len(node_data))
            # test_mask = index_to_mask(test_index, len(node_data))
            
            num_tr_nodes = int(len_data * Constants.ALPHA_TRAIN_PERCENT)
            print("num train nodes:",  num_tr_nodes, "num test nodes:", len_data - num_tr_nodes, "total:", len_data)
            train_mask = index_to_mask(random.sample(range(0, len_data-1), num_tr_nodes), len_data)
            test_mask = ~train_mask
            
            map_ids = {}
            map_index = 0                        # Maps java indices to [0, len) range
            for sample in node_data:
                xs.append(sample['fs'])
                ys.append(sample['label'])
                node_id = sample['id']
                node_ids.append(node_id)
                parent_id = sample['parent']
                ns = edge_data[str(parent_id)]
                if node_id not in map_ids:       # Map node_ids to [0, len) range
                    map_ids[node_id] = map_index
                    map_index += 1
                for n in ns:
                    if n not in map_ids:         # Map neighbour ids to [0, len) range
                        map_ids[n] = map_index
                        map_index += 1
                    if n != node_id:
                        from_nodes.append(map_ids[node_id])
                        to_nodes.append(map_ids[n])
                    
            x = torch.from_numpy(np.array(xs)).to(torch.float)       
            y = torch.from_numpy(np.array(ys)).to(torch.long)
            edge_index = torch.from_numpy(np.array([from_nodes, to_nodes])).to(torch.long)
            # edge_attr = torch.from_numpy(np.array(edge_weights)).to(torch.float)
            print('edge_index.max', edge_index.max())
            print('x.size(0)', x.size(0))
            
            data = Data(x=x, y=y, edge_index=edge_index)
            data.train_mask = train_mask
            data.test_mask = test_mask
            return True, data, map_ids
    return False, None, None
    
# ret_val, data, map_ids = read_alpha_node_data(Constants.ALPHA_NODE_FEATS_PATH, Constants.ALPHA_EDGES_PATH)
# print(map_ids)
dataset = AlphaNode(Constants.ALPHA_ROOT_PATH)
data = dataset[0]
print('num_nodes', data.num_nodes, 'num_classes', dataset.num_classes, 'dataset_len', len(dataset))
print('contains_self_loops', data.contains_self_loops())
print('contains_isolated_nodes', data.contains_isolated_nodes())
    
# Check there is only one graph
assert len(dataset) == 1

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                            dim=0), exact=True)
    data = gdc(data)

# Create network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                              normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                              normalize=not args.use_gdc)
        
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        
    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0]
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)
          
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
    
for epoch in range(1, 100):
    train()
    train_acc, test_acc = test()
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, test_acc))      
    
    
    
    
    
    
    
    
    
    
    
    
    