# -*- coding: utf-8 -*-
"""
Created on Fri May 22 02:19:00 2020

@author: melike
"""

import os


NODE_FILE_NAME = '68k_pixel_as_node.txt'
ALPHA_TRAIN_PERCENT = 0.8

""" ================================ Melike ================================ """

# BASE_PATH = "C:\\Users\\melike\\RS"

""" ================================== ROG ================================= """

BASE_PATH = "/home/rog/rs"

""" ======================================================================== """


ROOT_PATH = os.path.join(BASE_PATH, 'gcn', 'paths19', 'test_149')
VAI_PATH = os.path.join(BASE_PATH, 'vaihingen')
ALPHA_NODE_FEATS_PATH = os.path.join(VAI_PATH, 'alpha_feats_no_adj.txt')
ALPHA_EDGES_PATH = os.path.join(VAI_PATH, 'alpha_cliques.txt')
ALPHA_ROOT_PATH = os.path.join(VAI_PATH, '_pyg')


