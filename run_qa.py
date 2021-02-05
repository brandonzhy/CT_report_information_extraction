#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
import json
import os
import sys 

import argparse 
import numpy as np
import re

# Author: Huanyao Zhang
# Last update: 2020.10.29
# First create: 2019.03.23

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
from tqdm import tqdm
import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler 

from transformers import  BertTokenizer,AdamW,get_linear_schedule_with_warmup
from data.model_config import Config 

from models.bert_mrc import BertTagger
from data.data_utils import question_dic,subject2question,  relation2question, question_turn
from utils.evaluate_funcs import eval_checkpoint, get_indices_from_lable
from utils.file_utils import add_file_handler, get_logger



def args_parser():
    # start parser
    parser = argparse.ArgumentParser(description=' Trainer')

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run prediction")
    parser.add_argument("--cuda",  action='store_true')
    parser.add_argument("--data_dir", default=None, required=True, type=str, help="the input data dir")
    parser.add_argument("--config_path",type=str, required=False,
                        default="./pretrained_model/chinese-bert_chinese_wwm_pytorch/bert_config.json")
    parser.add_argument("--bert_model",
                        default="./pretrained_model/chinese-bert_chinese_wwm_pytorch", type=str, required=False,
                        help="bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--max_seq_length",
                        type=int, help="the maximum total input sequence length after ")
    parser.add_argument("--learning_rate", type=float, required=True,)
    parser.add_argument("--num_train_epochs", default=46, type=int, required=False,)
    parser.add_argument('--use_crf', action='store_true')
    parser.add_argument("--train_batch_size", default=32,  required=False,type=int)
    parser.add_argument("--output_dir", type=str, required=True, default="../bert_output")
    parser.add_argument("--loss_type", type=str, default="ce")

    parser.add_argument("--clip_grad", default=5.0, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)


    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--seed", type=int, default=3306)
    parser.add_argument('--torrence', type=int, default=5)

    parser.add_argument('--decode_mask', type=bool, default=True)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--use_rnn',  action='store_true')
    parser.add_argument('--sincos_position_embedding', action='store_true')
    parser.add_argument('--question_first',  action='store_true',
                       help="If true, concatenate question before question " )
    parser.add_argument('--start_end', action='store_true',
                        help="If true, use start and end labels to indicate position of entity  ")
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--classifier_sign", type=str, default="single_linear")
    args = parser.parse_args()




    return args

   


def get_features(config, logger):
    out_train_input_ids = os.path.join(config.data_dir, 'train_input_ids.npy')
    out_test_input_ids = os.path.join(config.data_dir, 'test_input_ids.npy')
    out_valid_input_ids = os.path.join(config.data_dir, 'valid_ids.npy')

    out_train_input_masks = os.path.join(config.data_dir, 'train_input_masks.npy')
    out_test_input_masks = os.path.join(config.data_dir, 'test_input_masks.npy')
    out_valid_input_masks = os.path.join(config.data_dir, 'valid_input_masks.npy')

    out_train_question_types = os.path.join(config.data_dir, 'train_question_types.npy')
    out_test_question_types = os.path.join(config.data_dir, 'test_question_types.npy')
    out_valid_question_types = os.path.join(config.data_dir, 'valid_question_types.npy')

    out_train_segment_ids = os.path.join(config.data_dir, 'train_segment_ids.npy')
    out_test_segment_ids = os.path.join(config.data_dir, 'test_segment_ids.npy')
    out_valid_segment_ids = os.path.join(config.data_dir, 'valid_segment_ids.npy')

    out_train_labels = os.path.join(config.data_dir, 'train_labels.npy')
    out_test_labels = os.path.join(config.data_dir, 'test_labels.npy')
    out_valid_labels = os.path.join(config.data_dir, 'valid_labels.npy')

    out_train_context_lengths = os.path.join(config.data_dir, 'train_context_lengths.npy')
    out_test_context_lengths = os.path.join(config.data_dir, 'test_context_lengths.npy')
    out_valid_context_lengths = os.path.join(config.data_dir, 'valid_context_lengths.npy')

    logger.info("train_input_id_filename :{}".format(out_train_input_ids))
    logger.info("test_input_id_filename :{}".format(out_test_input_ids))
    logger.info("valid_input_id_filename :{}".format(out_valid_input_ids))

    return (np.load(out_train_input_ids), np.load(out_valid_input_ids), np.load(out_test_input_ids),
            np.load(out_train_input_masks), np.load(out_valid_input_masks), np.load(out_test_input_masks),
            np.load(out_train_question_types), np.load(out_valid_question_types), np.load(out_test_question_types),
            np.load(out_train_segment_ids), np.load(out_valid_segment_ids), np.load(out_test_segment_ids),
            np.load(out_train_context_lengths), np.load(out_valid_context_lengths), np.load(out_test_context_lengths),
            np.load(out_train_labels), np.load(out_valid_labels), np.load(out_test_labels))


def load_data(config, logger):
    (train_input_ids, valid_input_ids, test_input_ids,
     train_input_mask, valid_input_mask, test_input_mask,
     train_question_types, valid_question_types, test_question_types,
     train_segment_ids, valid_segment_ids, test_segment_ids,
     train_context_lengths, valid_context_lengths, test_context_lengths,
     train_labels, valid_labels, test_labels
     ) = get_features(config, logger)
    train_input_ids = torch.tensor([f for f in train_input_ids], dtype=torch.long)
    train_input_mask = torch.tensor([f for f in train_input_mask], dtype=torch.long)
    train_segment_ids = torch.tensor([f for f in train_segment_ids], dtype=torch.long)
    train_context_lengths = torch.tensor([f for f in train_context_lengths], dtype=torch.long)
    train_question_types = torch.tensor([f for f in train_question_types], dtype=torch.long)

    train_label_ids = torch.tensor([f for f in train_labels], dtype=torch.long)
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids,
                               train_context_lengths, train_question_types)
    train_sampler = SequentialSampler(train_data)

    valid_input_ids = torch.tensor([f for f in valid_input_ids], dtype=torch.long)
    valid_input_mask = torch.tensor([f for f in valid_input_mask], dtype=torch.long)
    valid_segment_ids = torch.tensor([f for f in valid_segment_ids], dtype=torch.long)
    valid_context_lengths = torch.tensor([f for f in valid_context_lengths], dtype=torch.long)
    valid_question_types = torch.tensor([f for f in valid_question_types], dtype=torch.long)

    valid_label_ids = torch.tensor([f for f in valid_labels], dtype=torch.long)
    valid_data = TensorDataset(valid_input_ids, valid_input_mask, valid_segment_ids, valid_label_ids,
                               valid_context_lengths, valid_question_types)

    valid_sampler = SequentialSampler(valid_data)

    test_input_ids = torch.tensor([f for f in test_input_ids], dtype=torch.long)
    test_input_mask = torch.tensor([f for f in test_input_mask], dtype=torch.long)
    test_segment_ids = torch.tensor([f for f in test_segment_ids], dtype=torch.long)
    test_context_lengths = torch.tensor([f for f in test_context_lengths], dtype=torch.long)
    test_question_types = torch.tensor([f for f in test_question_types], dtype=torch.long)

    test_label_ids = torch.tensor([f for f in test_labels], dtype=torch.long)
    test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids,
                              test_context_lengths, test_question_types)

    test_sampler = SequentialSampler(test_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler,batch_size=config.train_batch_size, num_workers=4)

    dev_dataloader = DataLoader(valid_data, sampler=valid_sampler,batch_size=config.dev_batch_size, num_workers=4)

    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.test_batch_size, num_workers=4)

    num_train_steps = int(len(train_input_ids) / config.train_batch_size )* config.num_train_epochs
    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps


def load_model(config,  num_labels, device):
    model = BertTagger(config, num_labels, device)

    # prepare  optimzier
    param_optimizer = list(model.bert.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    
    #  The update of bias has nothing to do with weight decay
    if config.use_crf:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": model.crf_model.parameters(), "lr": config.learning_rate * 10}]
    else:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)


    return model, optimizer


def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader, config, \
          device, n_gpu, num_train_steps):
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    dev_best_acc = 0
    dev_best_precision = 0
    dev_best_recall = 0
    dev_best_f1 = 0
    dev_best_loss = 10000000000000

    logger = get_logger(config.output_dir)

    model_config = config.to_dict()
    logger.info('config  is :{}'.format(json.dumps(model_config, indent=2)))

    schedule = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=config.warmup_proportion*num_train_steps,
                                               num_training_steps=num_train_steps)

    logger.info(' length of training data is {}'.format(len(train_dataloader)))
    torrence_std = 0
    for idx in range(int(config.num_train_epochs)):
        model.train()
        tr_loss = 0

        nb_tr_examples, nb_tr_steps = 0, 0
        total_number_of_batch = len(train_dataloader)
        modulo = int(max(1, total_number_of_batch / 10))
        print('modulo = ', modulo)

        for step, batch in enumerate(train_dataloader):
            if config.start_end:
                input_ids, input_mask, segment_ids, label_starts, label_ends, context_length = batch
                label_ids = (label_starts, label_ends)
            else:
                input_ids, input_mask, segment_ids, label_ids, context_length, _ = batch

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            context_length = context_length.to(device)

            loss = model(input_ids, segment_ids, input_mask, label_ids, context_length, loss_sign=config.loss_type,
                         class_weight=None)
            tr_loss += loss.item()
            if n_gpu > 1:
                loss = loss.mean()
            optimizer.zero_grad()

            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)


            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            schedule.step()
            if nb_tr_steps % modulo == 0:
                # logger.info("-*-"*15)
                logger.info(
                    "epoch {} iter {}/{}- current training loss is : {}".format(idx + 1, step, total_number_of_batch,
                                                                                loss.item()))

        learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
        logger.info(
            "epoch {} done - training loss is : {} learning rate is {}".format(idx + 1, tr_loss / total_number_of_batch,
                                                                               learning_rate))

        tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader,
                                                                                           config, device, n_gpu,
                                                                                           eval_sign="dev")
        logger.info("......" * 10)
        logger.info("epoch {}  DEV: loss {} - acc {} - precision {} - recall {} - f1 {}".format(idx + 1, tmp_dev_loss,
                                                                                                tmp_dev_acc,
                                                                                                tmp_dev_prec,
                                                                                                tmp_dev_rec,
                                                                                                tmp_dev_f1))

        if tmp_dev_f1 > dev_best_f1:
            dev_best_acc = tmp_dev_acc
            dev_best_loss = tmp_dev_loss
            dev_best_precision = tmp_dev_prec
            dev_best_recall = tmp_dev_rec
            dev_best_f1 = tmp_dev_f1
            torrence_std = 0

            # export model
            if config.export_model:
                model_to_save = model.module if hasattr(model, "module") else model

                output_model_file = os.path.join(config.output_dir, "pytorch_model.pt")
                model_to_save.save(output_model_file)

        elif tmp_dev_f1 <= dev_best_f1:
            torrence_std += 1
            logger.info('torrence_std is {}  current dev score is {} ,best dev is {}'.format(torrence_std, tmp_dev_f1,
                                                                                             dev_best_f1))

        if torrence_std > config.torrence:
            break

    logger.info("=&=" * 15)
    tmp_train_loss, tmp_train_acc, tmp_train_prec, tmp_train_rec, tmp_train_f1 = eval_checkpoint(model,
                                                                                                 train_dataloader,
                                                                                                 config, device,
                                                                                                 n_gpu,
                                                                                                 eval_sign="train")
    logger.info(
        "train: loss {} - acc {} - precision {} - recall {} - f1 {}".format(tmp_train_loss,
                                                                            tmp_train_acc, tmp_train_prec,
                                                                            tmp_train_rec, tmp_train_f1))

    logger.info("best DEV:  loss {} - acc {} - precision {} - recall {} - f1 {} ".format(dev_best_loss, dev_best_acc,
                                                                                         dev_best_precision,
                                                                                         dev_best_recall, dev_best_f1))
    logger.info("=&=" * 15)

    if os.path.exists(os.path.join(config.output_dir, "pytorch_model.pt")):
        logger.info("......" * 10)
        logger.info('loading best model')
        best_model = model.load(os.path.join(config.output_dir, "pytorch_model.pt"))
        tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(best_model,
                                                                                                test_dataloader, config,
                                                                                                device, n_gpu,
                                                                                                eval_sign="test")
    else:
        logger.info("......" * 10)
        logger.info("loading current model")
        tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model, test_dataloader,
                                                                                                config, device, n_gpu,
                                                                                                eval_sign="test")

    logger.info("......" * 10)
    logger.info(
        "TEST: loss {} - acc {} - precision {} - recall {} - f1 {}".format(tmp_test_loss, tmp_test_acc, tmp_test_prec,
                                                                           tmp_test_rec, tmp_test_f1))
def get_index_number_from_logit(logit):
    index_number = []

    flag = 0
    for num in logit:
        if num != 0 and num%2==0:
            index_number.append(num)
 


    return index_number

def get_entity(logits,subject_type,relation_types=None):
    entity_list= []
    if relation_types:
         ject_lst = relation_types
    
    else :
                
        ject_lst = list(subject_type.keys())
        
    for ix  in range(len(ject_lst)):

        #logit_indice_list = get_index_number_from_logit(logits[ix])
        if len(set(logits[ix])) != 0:
            if not relation_types:
              #  entity_indices = get_indices_from_lable(logits[ix][1:],question_dic[ject_lst[ix]])
                entity_indices = get_indices_from_lable(logits[ix],question_dic[ject_lst[ix]])
            else:
                entity_indices = get_indices_from_lable(logits[ix],question_dic[relation_types[ix]])
                
            if entity_indices!= []:
                if relation_types:

                    entity_list.append({'type':relation_types[ix],'relation':relation_types[ix],'indices':entity_indices})
                else:
                    entity_list.append({'type':ject_lst[ix],'indices':entity_indices})

    return entity_list


def get_tail_entities(text,config,head_entity_token, device,subject_type,best_model,tokenizer):
    input_ids = []
    segment_ids = []
    input_mask = []
    context_lengths = []
    relation_types = []
    for relation in question_turn[subject_type]:
        rel_question = head_entity_token + relation2question[relation]
        encode_dic = tokenizer.encode_plus(text=[char for char in text],
                                           text_pair=[char for char in rel_question],
                                           max_length=config.max_seq_length,
                                           add_special_tokens=True, pad_to_max_length=True,
                                           truncation=True)
        input_ids.append(encode_dic['input_ids'])
        segment_ids.append(encode_dic['token_type_ids'])
        input_mask.append(encode_dic['attention_mask'])
        context_lengths.append([len(rel_question), len(text)])
        relation_types.append(relation)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    context_lengths = torch.tensor(context_lengths, dtype=torch.long, device=device)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)
    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    logits = best_model(input_ids, segment_ids, input_mask, labels=None,
                        context_lengths=context_lengths ,loss_sign=config.loss_type,
                         class_weight=None)
    tail_entity_dic_list = get_entity(logits, subject_type, relation_types=relation_types)
    return tail_entity_dic_list

def predict(model, config, device):
    print("......" * 10)
    print('loading best model')

    # load tokenizer and best model
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    
    text_file = [file for file in os.listdir(config.data_dir) if '.txt' in file]
    file_bar = tqdm(text_file)
    
    for filename in file_bar:
        file_bar.set_description("Processing {}".format(filename))
        res_predict = []
        with open(os.path.join(config.data_dir,filename), 'r', encoding='utf8') as f:
            data_text = f.readlines()
        for text in data_text:
            text = re.sub('\n','',text)
          
            if len(text)> 0 and ('肝' not in text) and ('胃' not in text) and ('甲状腺'not in text):
                input_ids = []
                
                segment_ids = []
                input_mask = []
                context_lengths = []
                text = re.sub('\n', '', text)
                res_dic = {'text': text}
                res_dic['spo_list'] = []
                
                for head_question in subject2question:
                    encode_dic = tokenizer.encode_plus(text=[char for char in text],
                                                       text_pair=[char for char in subject2question[head_question]],
                                                       max_length=config.max_seq_length,
                                                       add_special_tokens=True, pad_to_max_length=True,truncation=True)
                    input_ids.append(encode_dic['input_ids'])
                    segment_ids.append(encode_dic['token_type_ids'])
                    input_mask.append(encode_dic['attention_mask'])
                    context_lengths.append([len(head_question), len(text)])

                input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
                context_lengths = torch.tensor(context_lengths, dtype=torch.long, device=device)
                segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)
                input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
                with torch.no_grad():
                # consider all head_question as a batch
                    print('devices ={}'.format(device))    
                    logits = model(input_ids, segment_ids, input_mask, labels=None, context_lengths=context_lengths)
                    head_entity_dic_list = get_entity(logits, subject2question)
                    if head_entity_dic_list != []:
                        
                        for head_entity in head_entity_dic_list:
                            subject_type = head_entity['type']
                            if question_turn[subject_type]!=[]:

                                head_entity_indices = head_entity['indices']
                                head_entity_token_lst = []
                                for indice in head_entity_indices:

                                    head_entity_token = text[indice[0]:indice[1]]
                                    if len(head_entity_token)!=0 and head_entity_token not in head_entity_token_lst:  # a subject may have different mentions
                                        head_entity_token_lst.append(head_entity_token)

                               # subject_type = head_entity['type']
                            
                                for head_entity_token in head_entity_token_lst:
                                    tail_entity_dic_list = get_tail_entities(text,config,head_entity_token, device,subject_type,model,tokenizer)

                                    for tail_entity in tail_entity_dic_list:
                                        object_type = tail_entity['type']
                                        for indice in tail_entity['indices']:
                                            object_token = text[indice[0]:indice[1]]
                                            if len(object_token)>0:
                                                triple = {"subject_type": subject_type, "subject": head_entity_token,
                                                          "object_type": object_type, "object": object_token,
                                                          "relation": tail_entity['relation']}
                                                if triple not in res_dic['spo_list']:
                                                    res_dic['spo_list'].append(triple)
                            else:
                                indice =  head_entity['indices'][0]
                                subject = ''
                                if indice!=[]:
                                    subject = text[indice[0]:indice[1]]
                                
                                res_dic['spo_list'].append({"subject_type": subject_type, "subject": subject ,
                                                                  "object_type": "", "object": "",
                                                                  "relation":""})
                res_predict.append(res_dic)
        if not os.path.exists(os.path.join(config.output_dir,'prediction')):
        
            os.makedirs(os.path.join(config.output_dir,'prediction'))
        output_name = os.path.join(os.path.join(config.output_dir,'prediction'),filename.split('.')[0]+'.json')
        
        #print('output_name = {}'.format(output_name))
        
        with open(output_name, 'w', encoding='utf8') as f:
            for line in res_predict:
                f.write(json.dumps(line, ensure_ascii=False))
                f.write('\n')


def merge_config(args_config):
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    return model_config


def main():
    
    args_config = args_parser()
    print(args_config)
    config = merge_config(args_config)
   
   

   

    if config.cuda and  torch.cuda.is_available():
        device = torch.device("cuda:0")
        n_gpu = 1
    else:
        device = torch.device("cpu")
        n_gpu = 0
   
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    if config.do_train:
        logger = get_logger(config.output_dir)
        train_loader, dev_loader, test_loader, num_train_steps = load_data(config, logger)
        model, optimizer = load_model(config, num_train_steps, len(question_dic.keys()), device)
        train(model, optimizer, train_loader, dev_loader, test_loader, config, device, n_gpu, num_train_steps)
        
    if config.do_predict:
        model = BertTagger(config,  len(question_dic.keys()), device)
        #model_cpu = BertTagger(config,  len(question_dic.keys()), device)
        print('loading best model')
        model = model.load(os.path.join(config.output_dir, "pytorch_model.pt"))

        #pretained_dict  = model.state_dict()
        #model_cpu.load_state_dict(pretained_dict)
        #cpu_model = os.path.join(config.output_dir, "cpu")
        #output_model_file = os.path.join(cpu_model, "pytorch_model.pt")
        #model_cpu.save(output_model_file)
        
        predict(model, config, device)
       
    
if __name__ == "__main__":
    main()
