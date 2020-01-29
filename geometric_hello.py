#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:26:13 2020

@author: rog
"""

import torch
from torch_geometric.data import Data

"""
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)
"""

x = torch.tensor([[2,1], [5,6], [3, 7], [12, 0]], dtype=torch.float) # features per node,
                                                                     # index used for accessing nodes
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)                    # labels

edge_index = torch.tensor([[0, 0, 1, 2, 3],
                           [1, 3, 0, 1, 2]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)
print(data)
print("num edges:", data.num_edges)
print("num node features:", data.num_features) # This will be the number of attributes
print("num edge features:", data.num_edge_features)
print("num faces:", data.num_faces)