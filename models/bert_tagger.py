#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.03.23 
# First create: 2019.03.23 
# Description:
# BertTagger.py 



import os 
import sys

from transformers import BertForQuestionAnswering,BertModel

from utils.file_utils import load_big_file

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print("check the root_path")
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)



import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 



from layers.classifier import * 
# from models.bert_basic_model import *
from layers.bert_layernorm import BertLayerNorm


# class BertQuestionAnswering(nn.Module):
#     def __init__(self,config):
#         super(BertQuestionAnswering,self).__init__()
#         self.bert = BertForQuestionAnswering.from_pretrained(config.bert_model)
#



class BertTagger(nn.Module):
    def __init__(self, config, num_labels=4,load_best =False,dropout_sample=8):
        super(BertTagger, self).__init__()
        self.num_labels = num_labels
        self.load_best = load_best
        self.dropout_sample = dropout_sample
        self.config = config
        if load_best:
            self.bert = BertModel.from_pretrained(config.output_dir+'/model')
        else:
            self.bert = BertModel.from_pretrained(config.bert_model)

        # bert_config = BertConfig.from_dict(config.bert_config.to_dict())
        # self.bert = BertModel(bert_config)
        # self.bert = self.bert.from_pretrained(config.bert_model)
        # self.bert = BertForQuestionAnswering.from_pretrained(config.bert_model)

        # if config.bert_frozen == "true":
        #     print("!-!"*20)
        #     print("Please notice that the bert grad is false")
        #     print("!-!"*20)
        #     for param in self.bert.parameters():
        #         param.requires_grad = False

        self.hidden_size = config.hidden_size 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropouts = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob)for _ in range(8)])

        
        if config.classifier_sign == "single_linear":
            self.classifier = SingleLinearClassifier(config.hidden_size, self.num_labels) 
        elif config.classifier_sign == "multi_nonlinear":
            self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels) 
        else:
            raise ValueError 
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, \
        labels=None,context_lengths = None, loss_sign="ce", class_weight=None):
        batch_size = input_ids.shape[0]
        last_bert_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, \
            output_all_encoded_layers=False)
        last_bert_layer = last_bert_layer.view(batch_size,-1, self.hidden_size)
        # last_bert_layer = self.dropout(last_bert_layer)
        # 这里使用multi-dropout策略,使用多个dropout进行采样没然计算平均值，详见Multi-Sample Dropout for Accelerated Training and Better Generalization
        logit_samples = []
        for dropout in self.dropouts:
            logit_samples.append(self.classifier(dropout(last_bert_layer)))

        # logits = self.classifier(last_bert_layer)

        loss = 0
        len_total = 0
        # print("in bert_mrc ，logits = ",logits)


        '''
        step 1：先对一个batch中的所以段落计算头实体的loss，再对batch中的每个段落遍历所有的问题
        '''

        if labels is not None:
            if loss_sign == "ce":
                # loss = 0
                # len_total = 0
                loss_fct = CrossEntropyLoss()
                for logits in logit_samples:
                    # loss += loss_fct(logits, labels)
                    for qa,label,length in zip(logits,labels,context_lengths):
                        # print('in bert_mrc line 93: ',length)
                        # qa = torch.argmax(qa,dim = -1)
                        # 注意，这里传入CrossEntropyLoss的两个参数，第一个是predict:(seq_len,label_num)，第而个是label：（seq_len,)
                        # 传人（seq_len,)的会报错（rossEntropyLoss内部会有argmax操作）
                        # print('in bert_mrc line 94 shape of qa = ',qa.shape,'shape of label = ',label.shape)

                        loss += loss_fct(qa[length[0]+2:length[1]+length[0]+2], label[length[0]+2:length[1]+length[0]+2])

                    # loss += loss_fct(qa[:length.item()], label[:length.item()])
            else:
                print("DO NOT LOSS")

            return loss/(len(labels) + self.dropout_sample )
        else:
            # logit = logit_samples[0]
            # for ix in range(1,self.dropout_sample):
            #     logit.add(logit_samples[ix])
            #
            # return logit/self.dropout_sample
            return logit_samples[0]

    def _get_state_dict(self):
        model_state = {
            'state_dict':self.state_dict(),
            'config':self.config,
            'num_labels':self.num_labels,
            'load_best':self.load_best

        }
        # print('model_state = ',model_state)

        return model_state


    def save(self,model_file,pickle_protocal = 4):
        model_state = self._get_state_dict()

        torch.save(model_state,str(model_file),pickle_protocol=pickle_protocal )

    def load(cls,model_path):
        f = load_big_file(model_path)
        state = torch.load(f)
        model = cls._init_with_state_dict(state)
        model.eval()
        return model

    def _init_with_state_dict(cls,state):

        model = BertTagger(
            config=state['config'],
            num_labels=state['num_labels'],
            load_best=state['load_best']
        )

        model.load_state_dict(state_dict=state['state_dict'])

        return model




