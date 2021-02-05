#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

# Author: Huanyao Zhang
# Last update: 2020.10.23
# First create: 2019.03.23


import os 
import sys
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss ,BCELoss
from transformers import BertForQuestionAnswering,BertModel
from torchcrf import  CRF
import numpy as np
from utils.file_utils import load_big_file
import  torch.nn.functional as F
from layers.classifier import  SingleLinearClassifier, MultiNonLinearClassifier

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print("check the root_path")
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_

def pad_tensors(tensor_list,pad_length=168):
    ml = pad_length
    shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
    template = torch.zeros(*shape, dtype=torch.float)
    lens_ = [x.shape[0] for x in tensor_list]
    for i, tensor in enumerate(tensor_list):
        template[i, : lens_[i]] = tensor

    return template

class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
     #   pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
#        return self.dropout(self.pe[:x.shape[0], :])
        return self.dropout(self.pe)

class BertTagger(nn.Module):
    def __init__(self, config, num_labels=4,device = 'cpu',load_best =False,dropout_sample=8):
        super(BertTagger, self).__init__()
        self.num_labels = num_labels
        self.load_best = load_best
        self.dropout_sample = dropout_sample
        self.config = config
        self.use_rnn = config.use_rnn
        self.nlayers = config.nlayers
        self.hidden_size = 128
        self.rnn_type = 'LSTM'
        self.device =  device
        #self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.dropout_sample = dropout_sample
        if not config.start_end:
            if load_best:
                self.bert = BertModel.from_pretrained(config.output_dir)

            else:
                self.bert = BertModel.from_pretrained(config.bert_model)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.dropouts = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob)for _ in range(self.dropout_sample)])
            self.usr_crf = config.use_crf

            if  config.classifier_sign == "single_linear":
                self.classifier = SingleLinearClassifier(config.hidden_size, self.num_labels)
            elif config.classifier_sign == "multi_nonlinear":
                self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels)
            if self.use_rnn:
                self.rnn = getattr(torch.nn, self.rnn_type)(
                    config.hidden_size,
                    self.hidden_size,
                    num_layers=self.nlayers,
                    dropout=0.0 if self.nlayers == 1 else 0.5,
                    bidirectional=True,
                )


            if  self.usr_crf :
                self.crf_model = CRF(self.num_labels,batch_first=True)
        else:
            if load_best:
                self.bert = BertForQuestionAnswering.from_pretrained(config.output_dir+'/model')
            else:
                self.bert = BertForQuestionAnswering.from_pretrained(config.bert_model)
        self.to(self.device)



    def forward(self, input_ids, token_type_ids=None, attention_mask=None, \
        labels=None,context_lengths = None, loss_sign="ce", class_weight=None,threhold=0.3):
        batch_size = input_ids.shape[0]
        self.zero_grad()
        if self.config.start_end:
            start_score,end_score = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)

            start_score = F.softmax(start_score,dim = -1)
            end_score = F.softmax(end_score,dim=-1)

        else:
            last_bert_layer, pooled_output = self.bert(input_ids,attention_mask, token_type_ids )
            last_bert_layer = last_bert_layer.view(batch_size,-1, self.config.hidden_size).contiguous()

            last_bert_layer = torch.nn.LayerNorm(last_bert_layer.shape).to(self.device)(last_bert_layer)
            if self.use_rnn:
            
                lengths = torch.tensor([ length[0].item()+length[1].item() for length in context_lengths],dtype=torch.long).clamp(self.config.max_seq_length)-1
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    last_bert_layer, lengths, enforce_sorted=False,batch_first=True
                )

                # if initial hidden state is trainable, use this state

                try:
                    last_bert_layer, hidden = self.rnn(packed)
                except Exception as e:
                    print('last_bert_layer.shape = ,packed.shape = ,lengths = ',last_bert_layer.shape,packed.shape,lengths)
                else:
                    last_bert_layer, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                        last_bert_layer, batch_first=True
                    )
             
                    last_bert_layer = torch.nn.LayerNorm(last_bert_layer.shape)(last_bert_layer)
                    last_bert_layer = pad_tensors(last_bert_layer,self.config.max_seq_length)
            # 这里使用multi-dropout策略,使用多个dropout进行采样没然计算平均值，详见Multi-Sample Dropout for Accelerated Training and Better Generalization
            logit_samples = []
            for dropout in self.dropouts:
                logit_samples.append(self.classifier(dropout(last_bert_layer)))

        loss = 0
        len_total = 0

        if labels is not None:
            if self.config.use_crf:

                for logits in logit_samples:
                    assert  len(labels) == len(logits) == len(context_lengths)

                    loss += self.crf_model(logits.squeeze(-1),labels,attention_mask.to(torch.uint8))
                return -loss/(self.dropout_sample)
            else:
                if type(labels)==tuple:
                    label_start = labels[0]
                    label_end = labels[1]
                 
                    loss_fct = BCELoss()
              
                    for pre_start,pre_end,gold_start,gold_end ,length in zip(start_score,end_score,label_start,label_end,context_lengths):
                        if self.config.question_first:
                            pre_index = 2 + length[0]
                        else :
                            pre_index = 1
                        loss += loss_fct(pre_start[pre_index : pre_index + length[1]], gold_start[pre_index : pre_index + length[1]].float())
                        loss += loss_fct(pre_end[pre_index : pre_index+ length[1]], gold_end[pre_index : pre_index + length[1]].float())
                    return  loss/(2*len(context_lengths))
                else:

                    if loss_sign == "ce":

                        loss_fct = CrossEntropyLoss()
                        for logits in logit_samples:

                            for qa,label,length in zip(logits,labels,context_lengths):
                                

                                pre_index = 1
                                # only  calculate  masked  loss 
                                loss += loss_fct(qa[pre_index : pre_index + length[0]+ length[1]], label[pre_index : pre_index+ length[0]+ length[1]])


                        return loss/(len(labels) * self.dropout_sample )
                    else:
                        print("DO NOT LOSS")

        else:

                if self.config.use_crf:
                    logit = torch.tensor([lg[1:].tolist() for lg in logit_samples[0]],device=self.device)
                    if self.config.decode_mask:
                        attention_mask = torch.tensor([at[1:].tolist() for at in attention_mask ],device=self.device )
                        return self.crf_model.decode(logit.squeeze(-1),attention_mask.to(torch.uint8))
                    else:
                        return self.crf_model.decode(logit.squeeze(-1))
                else:
                    if self.config.start_end:
                      
                        return (start_score>threhold).long(),(end_score>threhold).long()
                    else:
                        return logit_samples[0]



    def _get_state_dict(self):
        model_state = {
            'state_dict':self.state_dict(),
            'config':self.config,
            'num_labels':self.num_labels,
            'load_best':self.load_best,
            'device':self.device,
            'dropout_sample':self.dropout_sample

        }

        return model_state

    def save(self,model_file,pickle_protocal = 4):
        model_state = self._get_state_dict()

        torch.save(model_state,str(model_file),pickle_protocol=pickle_protocal )

    def load(cls,model_path):
       
       # print('loading big file')  
        f = load_big_file(model_path)
     
       # print('loading big file finished')  
        #device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        device = torch.device('cpu')
        state = torch.load(f,map_location= device)
        model = cls._init_with_state_dict(state,device)
        
        model.eval()
        return model

    def _init_with_state_dict(cls,state,device):

        model = BertTagger(
            config=state['config'],
            num_labels=state['num_labels'],
            device=device,
            load_best=state['load_best'],
            dropout_sample = state['dropout_sample']
        )
       
        model.load_state_dict(state_dict=state['state_dict'])


        return model


