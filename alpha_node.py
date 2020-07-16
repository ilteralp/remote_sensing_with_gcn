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
from collections import Counter
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score, f1_score, precision_score

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import GCNConv
from torch.nn import Sequential, Linear
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
        node_file_path = osp.join(self.root, Constants.ALPHA_ADJ_NODE_FEATS_PATH)
        edge_file_path = osp.join(self.root, Constants.ALPHA_SPATIAL_ADJ_EDGES_PATH)
        
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

# Returns label percents in given mask with given labels
def get_label_percents(counts, labels, mask):
    uniq_labels = set(labels)
    mask_counts = dict.fromkeys(uniq_labels, 0)
    percents = {}
    for i, val in enumerate(mask):
        if val:
            mask_counts[labels[i]] = mask_counts[labels[i]] + 1
    for k in counts:
        percents[k] = mask_counts[k] / counts[k]
    
    return sorted(percents.items())

def print_train_test_info(labels, train_mask, test_mask):
    if len(train_mask) != len(labels) or len(train_mask) != len(test_mask):
        err_str = 'Different lengths, given %d vs %d vs %d' % (len(train_mask), len(labels), len(test_mask))
        raise ValueError(err_str)
    
    counts = dict(Counter(labels))
    for name, d in zip(['counts', 'train', 'test'], [sorted(counts.items()), get_label_percents(counts, labels, train_mask), get_label_percents(counts, labels, test_mask)]):
        print(name, '\t:', to_str(d))
    return counts
        
# Returns dict with float values in nicely formatted string
def to_str(d):
    res = "{"
    for pair in d:
        res += str(pair[0]) + ": "
        if isinstance(pair[1], int):
            res += str(pair[1]) + '\t'
        elif isinstance(pair[1], float):
            res += str("%.2f" % pair[1]) + '\t'
        else:
            print('Undefined type in pair!<')
    res += "}"
    return res

def create_masks(len_data):
    num_tr_nodes = int(len_data * Constants.ALPHA_TRAIN_PERCENT)
    train_ids = random.sample(range(0, len_data), num_tr_nodes)
    train_mask = index_to_mask(train_ids, len_data)
    test_mask = ~train_mask
    return train_mask, test_mask, num_tr_nodes, len_data-num_tr_nodes

def create_semisupervised_masks(len_data):
    num_tr_nodes = 140
    num_test_nodes = 1000
    train_ids = random.sample(range(0, len_data), num_tr_nodes)
    train_mask = index_to_mask(train_ids, len_data)
    tmp_mask = ~train_mask                                      # test mask will be created using this           
    tmp_ids = torch.arange(0, len_data)[tmp_mask]               # get tmp_ids
    shuffled_ids = torch.randperm(len(tmp_ids))                 # shuffle tmp_ids
    test_ids = tmp_ids[shuffled_ids[0:num_test_nodes]]          # get first num_test_nodes ids as test ids
    test_mask = index_to_mask(test_ids, len_data)
    return train_mask, test_mask, num_tr_nodes, num_test_nodes
    # print('train_mask', train_mask.sum().item())
    # print('test_mask', test_mask.sum().item())
    # cmul = torch.add(train_mask, test_mask)
    # print('cmul', cmul.sum().item())
        

def create_masks_for_clique_graph(node_data):
    train_end = int(len(node_data) * Constants.ALPHA_TRAIN_PERCENT)
    train_index = torch.arange(train_end, dtype=torch.long)
    test_index = torch.arange(train_end, len(node_data), dtype=torch.long)
    train_mask = index_to_mask(train_index, len(node_data))
    test_mask = index_to_mask(test_index, len(node_data))
    return train_mask, test_mask
        
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
            
            map_ids = {}
            map_index = 0                        # Maps java indices to [0, len) range
            num_skipped = 0
            for sample in node_data:
                node_id = sample['id']
                if str(node_id) in edge_data:
                    xs.append(sample['fs'])
                    ys.append(sample['label'])
                    node_ids.append(node_id)
                    # parent_id = sample['parent']
                    # ns = edge_data[str(parent_id)]
                    ns = edge_data[str(node_id)]
                    if node_id not in map_ids:       # Map node_ids to [0, len) range
                        map_ids[node_id] = map_index
                        map_index += 1
                    for n in ns:
                        if n not in map_ids:         # Map neighbour ids to [0, len) range
                            map_ids[n] = map_index
                            map_index += 1
                        from_nodes.append(map_ids[node_id])
                        to_nodes.append(map_ids[n])
                        # if n != node_id:           # Clique adjacency
                        #     from_nodes.append(map_ids[node_id])
                        #     to_nodes.append(map_ids[n])
                else:
                    num_skipped += 1
            print('Skipped %d nodes, using %d nodes' % (num_skipped,  len(node_data) - num_skipped))
            
            # for key, val in map_ids.items():
            #     print(key, 'became', val)
            
            len_data = len(node_data) - num_skipped
            train_mask, test_mask, num_tr_nodes, num_test_nodes = create_masks(len_data)
            # train_mask, test_mask, num_tr_nodes, num_test_nodes = create_semisupervised_masks(len_data)
            print("num train nodes:",  num_tr_nodes, "num test nodes:", len_data - num_tr_nodes, "total:", len_data)
            
            x = torch.from_numpy(np.array(xs)).to(torch.float)       
            y = torch.from_numpy(np.array(ys)).to(torch.long)
            edge_index = torch.from_numpy(np.array([from_nodes, to_nodes])).to(torch.long)
            # edge_attr = torch.from_numpy(np.array(edge_weights)).to(torch.float)
            counts = print_train_test_info(ys, train_mask, test_mask)

            data = Data(x=x, y=y, edge_index=edge_index)
            data.train_mask = train_mask
            data.test_mask = test_mask
            data.num_actual_classes = len(counts)
            return True, data, map_ids
    return False, None, None


def seed_everything(seed=4242):                                                  
 	random.seed(seed)                                                            
 	torch.manual_seed(seed)                                                      
 	torch.cuda.manual_seed_all(seed)                                             
 	np.random.seed(seed)                                                         
 	os.environ['PYTHONHASHSEED'] = str(seed)                                     
 	torch.backends.cudnn.deterministic = True                                    
 	torch.backends.cudnn.benchmark = False 

seed_everything()

# ret_val, data, map_ids = read_alpha_node_data(Constants.ALPHA_ADJ_NODE_FEATS_PATH, Constants.ALPHA_SPATIAL_ADJ_EDGES_PATH)
# print(map_ids)
dataset = AlphaNode(Constants.ALPHA_ROOT_PATH)
data = dataset[0]
if data.num_actual_classes.item() != dataset.num_classes:
    err_str = 'Number of dataset classes and actual classes are different!\n\t%d vs %d' % (dataset.num_classes, data.num_actual_classes.item())
    raise ValueError(err_str)
    
print('num_nodes', data.num_nodes, 'dataset_len', len(dataset))
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
        # self.lin = Sequential(Linear(10, 10))
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
model, data = Net().to(device), data.to(device)                         # Move network and data to device (CPU)
LR = 0.01
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=LR)

print('learning rate', LR)
          
def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask]) # y[mask] returns array of true values in mask
    # loss = F.mse_loss((model()[data.train_mask]).max(1)[1], data.y[data.train_mask])
    loss.backward() 
    optimizer.step()
    return loss
    
def compute_scores(y_test, y_pred):
    # conf = confusion_matrix(y_test, y_pred)
    # print('confusion matrix\n', conf)
    # print('balanced_accuracy', balanced_accuracy_score(y_test, y_pred))
    # print('precision_score', precision_score(y_test, y_pred, average="macro"))
    # print('recall_score', recall_score(y_test, y_pred, average='macro'))
    # print('f1_score', f1_score(y_test, y_pred, average="macro"))
    # recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average="macro")
    return f1

def labels_to_device(y_test, y_pred):
    if device.type == 'cpu':
        y_test = y_test.numpy()
        y_pred = y_pred.numpy()
    elif device.type == 'cuda':
        y_test = y_test.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    else:
        print('Unknown device:', device)
    return y_test, y_pred

@torch.no_grad()
def test():
    model.eval()                                                             # Sets the module in evaluation mode.
    logits, accs = model(), []                                               # Output of the model
    for name, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]                                        # Returns indices of max values in each row    
        y_test, y_pred = labels_to_device(data.y[mask], pred)
        f1 = compute_scores(y_test, y_pred)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()         # eq() computes element-wise equality.
        # accs.append(acc)
        accs.append(f1)
    return accs
    
for epoch in range(1, 200):
    loss = train()
    train_acc, test_acc = test()
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}, Loss: {:8.4f}'
    print(log.format(epoch, train_acc, test_acc, loss))      
    
    
    
    
    
    
    
    
    
    
    
    
    