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
# from torchcrf import CRF
from torchcrf import  CRF
import numpy as np
from utils.file_utils import load_big_file
import  torch.nn.functional as F

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print("check the root_path")
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)



import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss ,BCELoss


from data.data_utils import  entity_dic
from layers.classifier import * 
# from models.bert_basic_model import *
from layers.bert_layernorm import BertLayerNorm


# class BertQuestionAnswering(nn.Module):
#     def __init__(self,config):
#         super(BertQuestionAnswering,self).__init__()
#         self.bert = BertForQuestionAnswering.from_pretrained(config.bert_model)
#


START_TAG = entity_dic['<START>']
STOP_TAG = entity_dic['<STOP>']
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
    def __init__(self, config,device, num_labels=4,load_best =False,dropout_sample=1):
        super(BertTagger, self).__init__()
        self.num_labels = num_labels
        self.load_best = load_best
        self.dropout_sample = dropout_sample
        self.config = config
        self.use_rnn = config.use_rnn
        self.nlayers = config.nlayers
        self.hidden_size = 128
        self.rnn_type = 'LSTM'
        self.device = device
        self.position_embedding = PositionalEncoding(d_model=config.hidden_size,dropout=0.1, max_len=config.max_seq_length)
	#self.usr_crf = config.use_crf
        self.layerNorm = torch.nn.LayerNorm(torch.tensor([config.train_batch_size,config.max_seq_length,config.hidden_size]))
        if not config.start_end:
            if self.load_best:
                self.bert = BertModel.from_pretrained(config.output_dir+'/model')
                # self.bert = AlbertModel.from_pretrained(config.output_dir+'/model')


            else:
                self.bert = BertModel.from_pretrained(config.bert_model)
                # self.bert = AlbertModel.from_pretrained(config.bert_model)
            # self.hidden_size = config.hidden_size
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.dropouts = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob)for _ in range(dropout_sample)])
            

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
            # if  config.classifier_sign == "single_linear":
            #     self.classifier = SingleLinearClassifier(2*self.hidden_size, self.num_labels)
            # elif config.classifier_sign == "multi_nonlinear":
            #     self.classifier = MultiNonLinearClassifier(2*self.hidden_size, self.num_labels)


            if  self.config.use_crf :
                # self.transitions = torch.nn.Parameter(
                #     torch.randn(self.num_labels, self.num_labels)
                # )
                # self.transitions.detach()[START_TAG, :] = -10000
                # self.transitions.detach()[:, STOP_TAG] = -10000
                self.crf_model = CRF(self.num_labels,batch_first=True)
        else:
            if self.load_best:
                self.bert = BertForQuestionAnswering.from_pretrained(config.output_dir+'/model')
            else:
                self.bert = BertForQuestionAnswering.from_pretrained(config.bert_model)


        self.to(device)
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
        


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, \
        labels=None,context_lengths = None, question_type = None,loss_sign="ce", class_weight=None,threhold=0.3):
        batch_size = input_ids.shape[0]
        self.zero_grad()
        if self.config.start_end:
            start_score,end_score = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)

            start_score = F.softmax(start_score,dim = -1)
            end_score = F.softmax(end_score,dim=-1)

        else:
            position_ids = None
            if self.config.use_sincos_position:
                position_ids = torch.tensor([self.position_embedding(input_id).tolist() for  input_id in input_ids] ).to(self.device)
            last_bert_layer, pooled_output = self.bert(input_ids,
                                                        attention_mask,
                                                        token_type_ids,
                                                        position_ids
                                                        )
            last_bert_layer = torch.nn.LayerNorm(last_bert_layer.shape).to(self.device)(last_bert_layer)
            # last_bert_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
          #  if last_bert_layer.shape[0] != self.config.train_batch_size:
             #   last_bert_layer_tmp = last_bert_layer
             #   for _ in range(last_bert_layer.shape[0],self.config.train_batch_size):
                    
               #     last_bert_layer_tmp = torch.cat((last_bert_layer_tmp,last_bert_layer))
             #   last_bert_layer = last_bert_layer_tmp
         #   last_bert_layer = last_bert_layer.view(self.config.train_batch_size,-1, self.config.hidden_size).contiguous()
           # print(last_bert_layer.device)
         #   print(last_bert_layer.shape)
            #if last_bert_layer.shape[1]== self.config.max_seq_length:
              #  last_bert_layer =self.layerNorm(last_bert_layer)

            if self.use_rnn:
                # last_bert_layer = self.rnn(last_bert_layer)
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
                    # print('shape of last_bert_layer =',last_bert_layer.shape )
                    last_bert_layer = torch.nn.LayerNorm(last_bert_layer.shape)(last_bert_layer)
                    last_bert_layer = pad_tensors(last_bert_layer,self.config.max_seq_length)
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
            if self.config.use_crf:
                score = 0

                labels_lst = []
                logits_lst = []
                length_lst = []
             # pad tags if using batch-CRF decoder
                for logits in logit_samples:
                    # print('logits.shape = {}, labels.shape = {},context_length.shape = {}'.format(logits.shape,labels.shape,context_lengths.shape))
                    assert  len(labels) == len(logits) == len(context_lengths)
                    if class_weight:
                        attention_mask = attention_mask.to(torch.uint8)
                        #logits = logits.squeeze(-1) 
                        for ix in range(logits.shape[0]):
                            #print('labels[ix].shape ={} ,labels[ix].unsqueeze(0).shape =  {},labels[ix] = {}'.format(labels[ix].shape,labels[ix].unsqueeze(0).shape,labels[ix]))
                            loss += class_weight[question_type[ix].item()]*self.crf_model(logits[ix].unsqueeze(0),labels[ix].unsqueeze(0),attention_mask[ix].unsqueeze(0))
                        loss /=  logits.shape[0]
                    else:

                        loss += self.crf_model(logits.squeeze(-1),labels,attention_mask.to(torch.uint8))

                return -loss/(self.dropout_sample)
            else:
                if type(labels)==tuple:
                    label_start = labels[0]
                    label_end = labels[1]
                    # print('start_end')
                    # ignored_index = start_score.shape[1]
                    # start_score = F.softmax(start_score,dim = -1)
                    # end_score = F.softmax(end_score,dim=-1)
                    # start_score.clamp_(0, 1)
                    # end_score.clamp_(0, 1)


                    loss_fct = BCELoss()
                            # print('in bert_mrc line 93: ',length)
                            # qa = torch.argmax(qa,dim = -1)
                            # 注意，这里传入CrossEntropyLoss的两个参数，第一个是predict:(seq_len,label_num)，第而个是label：（seq_len,)
                            # 传人（seq_len,)的会报错（rossEntropyLoss内部会有argmax操作）
                            # print('in bert_mrc line 94 shape of qa = ',qa.shape,'shape of label = ',label.shape)
                    for pre_start,pre_end,gold_start,gold_end ,length in zip(start_score,end_score,label_start,label_end,context_lengths):

                        # label_start_indices = np.where(label_start > threhold)[0]
                        # label_end_indices = np.where(label_end > threhold)[0]
                        loss += loss_fct(pre_start[2 + length[0] : 2 + length[0] + length[1]], gold_start[2 + length[0] : 2 + length[0] + length[1]].float())
                        loss += loss_fct(pre_end[2 + length[0] : 2 + length[0] + length[1]], gold_end[2 + length[0] : 2 + length[0] + length[1]].float())
                    return  loss/(2*len(context_lengths))
                else:

                    if loss_sign == "ce":
                        # loss = 0
                        # len_total = 0
                        loss_fct = CrossEntropyLoss()
                        for logits in logit_samples:
                           # print("context_lengths={},question_type={}".format(context_lengths,question_type))
                            for qa,label,length,qa_type in zip(logits,labels,context_lengths,question_type):
                                # print('in bert_mrc line 93: ',length)
                                # qa = torch.argmax(qa,dim = -1)
                                # 注意，这里传入CrossEntropyLoss的两个参数，第一个是predict:(seq_len,label_num)，第而个是label：（seq_len,)
                                # 传人（seq_len,)的会报错（rossEntropyLoss内部会有argmax操作）
                                # print('in bert_mrc line 94 shape of qa = ',qa.shape,'shape of label = ',label.shape)
                                # loss += loss_fct(qa, label)
                                if self.config.question_first:
                                    pre_index = 2 + length[0]
                                else:
                                    pre_index = 1
                               #question before context
                               # pre_index = 2 + length[0]
                                if class_weight:
                                    loss += class_weight[qa_type.item()]*loss_fct(qa[pre_index : pre_index + length[1]], label[pre_index: pre_index + length[1]])
                                else:
                                    loss += loss_fct(qa[pre_index : pre_index + length[1]], label[pre_index: pre_index + length[1]])

                            # loss += loss_fct(qa[:length.item()], label[:length.item()])

                        return loss/(len(labels) * self.dropout_sample )
                    else:
                        print("DO NOT LOSS")

        else:
                # logit = logit_samples[0]


                # for ix in range(1,self.dropout_sample):
                #     logit.add(logit_samples[ix])
                #
                # return logit/self.dropout_sample
                if self.config.use_crf:
                   # print('use viterbi_code')
                    tags = []
                    # print(context_lengths)
                    # for feats,length in zip(logit_samples[0],context_lengths.detach().numpy().tolist()):
                    #     confidences, tag_seq, scores = self._viterbi_decode(feats[2 + length[0] : 2 + length[0] + length[1]])
                    #     tags.append(tag_seq)
                    return self.crf_model.decode(logit_samples[0].squeeze(-1),attention_mask.to(torch.uint8))
                    # return self.crf_model.decode(logit_samples[0].squeeze(-1),attention_mask.to(torch.uint8))
                else:
                    if self.config.start_end:
                        print(start_score[0],'\n',end_score[0])
                        return (start_score>threhold).long(),(end_score>threhold).long()
                    else:
                        return logit_samples[0]




    def _get_state_dict(self):
        model_state = {
            'state_dict':self.state_dict(),
            'config':self.config,
            'num_labels':self.num_labels,
            'load_best':self.load_best,
            'device':self.device

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
            device=state['device'],
            num_labels=state['num_labels'],
            load_best=state['load_best']
	
        )

        model.load_state_dict(state_dict=state['state_dict'])

        return model


    def _viterbi_decode(self, feats):
        backpointers = []
        backscores = []
        scores = []

        init_vvars = (
            torch.FloatTensor(1, self.num_labels).fill_(-10000.0)
        )
        init_vvars[0][START_TAG] = 0
        forward_var = init_vvars

        for feat in feats:
            next_tag_var = (
                    forward_var.view(1, -1).expand(self.num_labels, self.num_labels)
                    + self.transitions
            )
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = (
                forward_var
                + self.transitions[STOP_TAG]
        )
        terminal_var.detach()[STOP_TAG] = -10000.0
        terminal_var.detach()[START_TAG] = -10000.0
        best_tag_id = np.argmax(terminal_var.unsqueeze(0))

        best_path = [best_tag_id]

        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())
            scores.append([elem.item() for elem in softmax.flatten()])

        start = best_path.pop()
        assert start == START_TAG
        best_path.reverse()

        for index, (tag_id, tag_scores) in enumerate(zip(best_path, scores)):
            if type(tag_id) != int and tag_id.item() != np.argmax(tag_scores):
                swap_index_score = np.argmax(tag_scores)
                scores[index][tag_id.item()], scores[index][swap_index_score] = (
                    scores[index][swap_index_score],
                    scores[index][tag_id.item()],
                )
            elif type(tag_id) == int and tag_id != np.argmax(tag_scores):
                swap_index_score = np.argmax(tag_scores)
                scores[index][tag_id], scores[index][swap_index_score] = (
                    scores[index][swap_index_score],
                    scores[index][tag_id],
                )

        return best_scores, best_path, scores

    def _score_sentence(self, feats, tags, lens_):

        start = torch.tensor(
            [START_TAG]
        )
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.tensor(
            [STOP_TAG]
        )
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, 2+lens_[i][0]+lens_[i][1] :] = STOP_TAG

        score = torch.FloatTensor(feats.shape[0])

        for i in range(feats.shape[0]):
            # r = torch.LongTensor(range(lens_[i][1]))

            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, lens_[i][0]+2:lens_[i][0] + lens_[i][1]+2], pad_start_tags[i, lens_[i][0]+2: lens_[i][0] +lens_[i][1]+2]
                ]
            ) + torch.sum(feats[i, lens_[i][0]+2: lens_[i][0] + lens_[i][1]+2, tags[i, lens_[i][0]+2: lens_[i][0] + lens_[i][1]+2]])

        return score

    def _forward_alg(self, feats, lens_):

        init_alphas = torch.FloatTensor(self.num_labels).fill_(-10000.0)
        init_alphas[START_TAG] = 0.0
        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float
            )

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)

        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = (
                    emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                    + transitions
                    + forward_var[:, i, :][:, :, None]
                    .repeat(1, 1, transitions.shape[2])
                    .transpose(2, 1).contiguous()
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned
        lens_ = [2+length[0]+length[1] for length in lens_]
        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        terminal_var = forward_var + self.transitions[
                                         STOP_TAG
                                     ][None, :].repeat(forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha


