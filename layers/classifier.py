#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Huanyao Zhang
# Last update: 2020.10.29
# First create: 2019.03.23

import os 
import sys 

import torch

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)





class SingleLinearClassifier(torch.nn.Module):
    def __init__(self, hidden_size, num_label):
        super(SingleLinearClassifier, self).__init__()
        self.num_label = num_label 
        self.classifier = torch.nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        features_output = self.classifier(input_features)

        return features_output 


class MultiNonLinearClassifier(torch.nn.Module):
    def __init__(self, hidden_size, num_label):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label 
        self.classifier1 = torch.nn.Linear(hidden_size, int(hidden_size / 2))
        self.classifier2 = torch.nn.Linear(int(hidden_size / 2), num_label)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = torch.nn.ReLU()(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

