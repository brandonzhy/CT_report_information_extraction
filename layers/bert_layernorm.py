#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# Author: Huanyao Zhang
# Last update: 2020.10.29
# First create: 2019.03.23



import os 
import sys 
import copy 
import json 
import math 
import logging 
import tarfile 
import tempfile 
import shutil 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)



import torch 
from torch import nn 
from torch.nn import CrossEntropyLoss 



class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        # construct a layernorm module in the TF style
        # epsilon inside the square are not 
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps 


    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias 