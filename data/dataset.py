#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Huanyao Zhang
# Last update: 2020.10.23
# First create: 2019.03.23

from torch.utils.data import Dataset
class QADataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    # def __init__(self, input_ids, input_masks, segment_ids, start_ids,end_ids,context_lengths,context,question_types):
    def __init__(self, input_ids, input_masks, segment_ids, label_ids,context_lengths,question_types):

        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lable_ids = label_ids
        # self.start_ids = start_ids
        # self.end_ids = end_ids

        self.context_lengths = context_lengths
        # self.context = context
        self.question_types =question_types


    def __getitem__(self, index):
        input_id = self.input_ids[index]
        input_mask = self.input_masks[index]
        segment_id = self.segment_ids[index]
        # start_ids = self.start_ids[index]
        # end_ids = self.end_ids[index]
        label = self.lable_ids[index]
        context_length = self.context_lengths[index]
        # context = self.context[index]
        question_type = self.question_types[index]

        # return tuple([input_id,input_mask,segment_id,start_ids,end_ids,context_length,context,question_type])

        return tuple([input_id,input_mask,segment_id,label,context_length,question_type])

    def __len__(self):
        return self.input_ids.size(0)


class QADataset_SE(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    # def __init__(self, input_ids, input_masks, segment_ids, start_ids,end_ids,context_lengths,context,question_types):
    def __init__(self, input_ids, input_masks, segment_ids, start_labels,end_labels,context_lengths,context,question_types):

        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.start_labels = start_labels
        self.end_labels = end_labels
        # self.start_ids = start_ids
        # self.end_ids = end_ids

        self.context_lengths = context_lengths
        self.context = context
        self.question_types =question_types


    def __getitem__(self, index):
        input_id = self.input_ids[index]
        input_mask = self.input_masks[index]
        segment_id = self.segment_ids[index]
        # start_ids = self.start_ids[index]
        # end_ids = self.end_ids[index]
        start_label = self.start_labels[index]
        end_label = self.end_labels[index]
        context_length = self.context_lengths[index]
        context = self.context[index]
        question_type = self.question_types[index]

        # return tuple([input_id,input_mask,segment_id,start_ids,end_ids,context_length,context,question_type])

        return tuple([input_id,input_mask,segment_id,start_label,end_label,context_length,context,question_type])

    def __len__(self):
        return self.input_ids.size(0)