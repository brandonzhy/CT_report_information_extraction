#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

# Author: Huanyao Zhang
# Last update: 2020.10.23
# First create: 2019.03.23

# data_utils.py
import json
import os
import sys 
import csv 
import logging 
import argparse 
import random 
import numpy as np
import pandas as pd
import re
import torch
from tqdm import tqdm, trange
from transformers import BertTokenizer
# from data.data_process.prepare_data import  rel_question_dic,rel_question_dic_key_words,rel_question_dic_questions
# root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
# if root_path not in sys.path:
#     sys.path.insert(0, root_path)
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler

relation2question = {
    'Location': '位置在哪',
    'Shape': '形状如何',
    'Size': '大小是多少',
    'Density': '是否实性或有磨玻璃',
    'Lymph': '与淋巴结是否有关系',
    'Pleura': '是否侵犯胸膜',
    'Bronchus': '是否侵犯支气管',
    'ChestWall': '是否侵犯胸壁',
    'Hilus': '是否侵犯肺门',
    'Vessel': '是否侵犯血管',
    'Intension': '是否有高代谢或强化',
    'PulmonaryAtelectasis': '是否伴有肺阻塞或肺不张',
    'Esophagus': '是否侵犯食管',
}
question_dic = {

    '<unk>': 0,
    'O': 1,
    'Mass': 2,
    'Location': 4,
    'Shape': 6,
    'Size': 8,
    'Density': 10,
    'Lymph': 12,
    'Pleura': 14,
    'Bronchus': 16,
    'ChestWall': 18,
    'Hilus': 20,
    'Vessel': 22,
    'Intension': 24,
    'PulmonaryAtelectasis': 26,
    'Esophagus': 28,
    'Effusion': 30,
    'Spinal': 32,
    'Rib': 34,

    'Mass-I': 3,
    'Location-I': 5,
    'Shape-I': 7,
    'Size-I': 9,
    'Density-I': 11,
    'Lymph-I': 13,
    'Pleura-I': 15,
    'Bronchus-I': 17,
    'ChestWall-I': 19,
    'Hilus-I': 21,
    'Vessel-I': 23,
    'Intension-I': 25,
    'PulmonaryAtelectasis-I': 27,
    'Esophagus-I': 29,
    'Effusion-I': 31,
    'Spinal-I': 33,
    'Rib-I': 35,
}
id2relation = {
   2:'Mass',
    4: 'Location',
    6: 'Shape',
    8: 'Size',
    10: 'Density',
    12:'Lymph',
    14: 'Pleura',
    16: 'Bronchus',
    18: 'ChestWall',
    20: 'Hilus',
    22: 'Vessel',
    24: 'Intension',
    26: 'PulmonaryAtelectasis',
    28: 'Esophagus',
    30: 'Effusion',
    32: 'Spinal',
    34: 'Rib',
}

question_turn = {

    'Mass': [
        'Location',
        'Shape',
        'Size',
        'Density',
        'Pleura',
        'Bronchus',
        'ChestWall',
        'Hilus',
        'Vessel',
        'Intension',
        'PulmonaryAtelectasis',
        'Esophagus'],
    'Lymph': ['Location', 'Size'],
    'Pleura': ['Location'],
    'Effusion': ['Location']
}

subject2question = {
    'Mass': '结节或肿物或肿块或灶或病变或占位征象的描述',
    'Lymph': '淋巴结的描述',
    'Pleura': '左右侧胸膜的描述',
    'Effusion': '胸水的描述',

}

id2subject = {
    2: 'Mass',
    4: 'Lymph',
    6: 'Pleura',
    8: 'Effusion'

}

entity2question= {
'position':'职业或职位的描述',
'movie':'电影',
'company':'公司名称',

'book':'书籍',
'address':'位于哪或在哪',
'game':'游戏名称',
'government':'政府机构或会',
    'organization':'组织名称或队伍名称',
    'mobile':'联系电话号码',
    'name':'姓名',
    'scene':'著名景点',
}
entity2id = {
        'Others':0,
        '<SPEC>':1,
        'position': 2,
        'movie': 4,
        'address': 6,
        'book': 8,
        'government': 10,
        'game': 12,
        'company': 14,
        'organization': 16, 
        'mobile': 18,
        'scene': 20,
        'name':22,
        'email':24,
    
        'I-position': 3,
        'I-movie': 5,
        'I-address': 7,
        'I-book': 9,
        'I-government': 11,
        'I-game': 13,
        'I-company': 15,
        'I-organization': 17,
        'I-mobile': 19,
      'I-scene': 21,
      'I-name': 23,
      'I_email':25,
    }
id2entity = {
    2: 'position',
    4: 'movie',
    6: 'address',
    8: 'book',
    10:'government',
    12:'game',
    14:'company',
    16:'organization',
    18:'mobile',
    20:'scene',
    22:'name',
    24:'email'
}
def clean_data(text):
    """
    1、替换为空格的：
    (/topics/zh-cn/17)
    targetUrl=http://cochraneclinicalanswers.com/doi/10.1002/cca.355/full)
    (http://www.cdc.gov/ncezid/dvbd/about.html)

    2、英文的括号替换为中文的（为了其他其他的匹配）
    3、空格替换
    SPO中的subject和object也要空格替换
    text 中的空格替换为，

    """

    text = re.sub(r'[(]', '（', text)
    text = re.sub(r'[)]', '）', text)
    text = re.sub(r',', "，", text)

    text = re.sub(r'（/.*）', '', text)
    text = re.sub(r'（http.*）', '', text)
    text = re.sub('targetUrl=http.*）', '', text)
    text = re.sub(r'[#*\s]{2,}', '-', text)
    text = re.sub(r'[，]{2,}', '，', text)

    return text




def parse_triple_to_qa_directly(data,num_neg_sample = 10):

    text = data['text']
    text = clean_data(text)
    dataset = []
    neg_samples = []
    pos_samples = []
    data_line = {}


    for spo in data['spo_list']:
        subject_type = spo['subject_type']
        object_type = spo['object_type']['@value']
        subject = re.sub(r',', "，",spo['subject'])
        subject = re.sub(r'[(]', '（', subject)
        subject = re.sub(r'[)]', '）', subject)
        object = re.sub(r',', "，",spo['object']['@value'])
        object = re.sub(r'[(]', '（', object)
        object = re.sub(r'[)]', '）', object)
        relation = spo['predicate']

        if subject_type in question_turn  and subject_type=='疾病':
            if  subject_type not in data_line:
                data_line[subject_type] = {}
                for sub_rel in question_turn[subject_type]:
                    data_line[subject_type][sub_rel] = []

            try:
                subject_index =  text.find(subject)

            except Exception as  e:
                print('e = {},subject ={},text ={}'.format(e,subject,text))

            data_line[subject_type][subject_type]= [subject_index,subject_index + len(subject)]

            for sub_rel in question_turn[subject_type]:
                if sub_rel == relation :
                    try:
                        object_index = text.find(object)
                    except Exception as e:
                        print('e = {},object ={},text ={}'.format(e,object,text))
                    # if  data_line[subject_type][relation]!=[]:
                    index = [object_index,object_index + len(object)]
                    if index not in data_line[subject_type][relation]:
                        data_line[subject_type][relation].append(index)

                    # if len(data_line[subject_type][relation])>1:
                    #     print('text = {},relation = {}'.format(text,relation))
                else:
                    data_line[subject_type][sub_rel] = []

    for subject_type in data_line:
        dataset.append({
            'context': text,
            'subject_type': subject_type,
            'label': [subject_index, subject_index + len(subject)]
        })
        pos_samples.append({
            'context': text,
            'subject_type': subject_type,
            'label': [subject_index, subject_index + len(subject)]
        })
        num_pos = 0
        for rel in data_line[subject_type]:
            if rel!=subject_type:
                if data_line[subject_type][rel] != []:
                    num_pos += 1
                    dataset.append({
                        'context': text,
                        'relation_type': rel,
                        'subject_token': subject,
                        'object_token':[text[index[0]:index[1]] for index in data_line[subject_type][rel] ],
                        'object_type':object_type,
                        'label': data_line[subject_type][rel]
                    })
                    pos_samples.append({
                        'context': text,
                        'relation_type': rel,
                        'subject_token': subject,
                        'object_token': [text[index[0]:index[1]] for index in data_line[subject_type][rel]],
                         'object_type':object_type,
                        'label': data_line[subject_type][rel]
                    })

                else:
                    neg_samples.append({
                        'context': text,
                        'relation_type': rel,
                        'subject_token': subject,
                        'object_token': [text[index[0]:index[1]] for index in data_line[subject_type][rel]],
                         'object_type':object_type,
                        'label': data_line[subject_type][rel]
                    })
        len_neg_samples = len(neg_samples)
        if len_neg_samples <=  num_neg_sample:
            choice_indexs = list(range(len_neg_samples))

        else:
            choice_indexs = np.random.choice(len_neg_samples,num_neg_sample)

        for choice in choice_indexs :
                dataset.append(neg_samples[choice])

        pos_sampling_rate = int(min(len_neg_samples,num_neg_sample)/len(pos_samples)/2)  if len(pos_samples) > 0  else 0
        for ix in range(pos_sampling_rate):
            for sample in pos_samples:
                dataset.append(sample)

    return dataset

def add_neg_samples(filename):
    dataset= []
    neg_samples = []
    pos_samples = []
    data = json.load(open(filename, 'r', encoding='utf8') )
    for context ,question_ans in data.items():
        subject_token = ''
        subject_type = ''
        object_list = []
        for questions, ans in question_ans.items():
            label = []
            if (type(ans[0])!=list) :
                label = [ans[0], ans[1]]
                entity_type = ans[-1]

            else :
                entity_type =ans[0][-1]
                label = [[a[0], a[1]] for a in ans]
                ans = ans[0]
            if entity_type in question_turn:
                subject_type = entity_type
                subject_token = context[ans[0]:ans[1]]
                pos_samples.append(
                    {
                        "context": context,
                        "subject_type": entity_type,
                        "subject_token": subject_token,

                        "label": label
                    }
                )
                dataset.append(
                    {
                        "context": context,
                        "subject_type": entity_type,
                        "subject_token": subject_token,
                        "label": label
                    }
                )

            else :
                dataset.append({
                    "context": context,
                    "relation_type": entity_type,
                    "subject_token": subject_token,
                    "object_type": entity_type,
                    "label": label
                })
                object_list.append(entity_type)
                pos_samples.append({
                    "context": context,
                    "relation_type": entity_type,
                    "subject_token": subject_token,
                    "object_type": entity_type,
                    "label": label
                })


        if subject_type in question_turn:
            for object in question_turn[subject_type]:
                if object not in object_list:
                    neg_samples.append(
                        {
                            "context": context,
                            "relation_type": object,
                            "subject_token": subject_token,
                            "object_type": object,
                            "label": []
                        }
                    )
                    dataset.append(
                        {
                            "context": context,
                            "relation_type": object,
                            "subject_token": subject_token,
                            "object_type": object,
                            "label": []
                        }
                    )
        if len(neg_samples)>2*len(pos_samples):
            for sample in pos_samples:
                dataset.append(sample)

        with open(filename.split('.')[0]+'_neg_format.json','w',encoding='utf8') as f:
            json.dump(dataset,f,indent=4,ensure_ascii=False)



def convert_triple_to_qa_format_directly(filename,mode,num_neg_sample = 10):

    dataset = []

    if mode !='test':
        with open(filename,'r',encoding='utf8') as f:
            for line in f:
                data = json.loads(line)
                dataset.extend(parse_triple_to_qa_directly(data,mode=mode,num_neg_sample=num_neg_sample))

        if 'train' in filename:
            index = int(0.8*len(dataset))
            dataset_train = dataset[:index]
            dataset_valid = dataset[index:]
            # dataset_train_20= dataset_train[:int(0.05*len(dataset_train))]
            # dataset_valid_20= dataset_valid[:int(0.05*len(dataset_valid))]

            with open( os.path.join(os.path.split(filename)[0] ,'train_qa_disease.json'),'w',encoding='utf8') as f:
                json.dump(dataset_train,f,indent=4,ensure_ascii=False)
            with open( os.path.join(os.path.split(filename)[0] ,'valid_qa_disease.json'),'w',encoding='utf8') as f:
                # for data in dataset_valid:
                json.dump(dataset_valid,f,indent=4,ensure_ascii=False)

            # with open( os.path.join(os.path.split(filename)[0] ,'train_qa_disease_5.json'),'w',encoding='utf8') as f:
            #     json.dump(dataset_train_20,f,indent=4,ensure_ascii=False)
            # with open( os.path.join(os.path.split(filename)[0] ,'valid_qa_disease_5.json'),'w',encoding='utf8') as f:
            #     # for data in dataset_valid:
            #     json.dump(dataset_valid_20,f,indent=4,ensure_ascii=False)

        elif 'dev' in filename :

            with open(os.path.join(os.path.split(filename)[0] ,'test_qa_disease.json'), 'w', encoding='utf8') as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            # dataset_20 = dataset[:int(0.05*len(dataset))]
            # with open(  os.path.join(os.path.split(filename)[0] ,'test_qa_disease_5.json'), 'w', encoding='utf8') as f:
            #     json.dump(dataset_20, f, indent=4, ensure_ascii=False)


    else:
        with open(filename,'r',encoding='utf8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append( parse_triple_to_qa_directly(data,mode=mode))
        output_name = filename.split('.')[0] + '_qa.json'

        with open(output_name,'w',encoding='utf8') as f:
            # for data in dataset:
            json.dump(dataset,f,indent=4,ensure_ascii=False)


def convert_to_input_feature(filename,tokenizer,max_seq_length=320):
    features= []
    input_ids = []
    segment_ids = []
    input_mask = []
    labels = []
    context_lengths = []
    question_types = []
    data = json.load(open(filename,'r',encoding='utf8'))
    for line in data:
        # if line['label']!=[]:
        context = line['context']
        if len(context) < max_seq_length - 20:
            if 'subject_type' in line:
                subject_type =line['subject_type']
                subject_question = subject2question[subject_type]
                encode_dic = tokenizer.encode_plus(text=[char for char in context],
                                                   text_pair=[char for char in subject_question],
                                                   max_length=max_seq_length,
                                                   add_special_tokens=True, pad_to_max_length=True,truncation=True)
                label = np.zeros(max_seq_length, dtype=np.long)
                ans = line['label']
                pre_idnex = 1
                if type(ans[0]) == list:
                    for a in ans:
                        label[pre_idnex + a[0]] = question_dic[subject_type]
                        label[1+ pre_idnex + a[0]:pre_idnex + a[1]] = question_dic[subject_type] + 1
                else:
                    label[pre_idnex + ans[0]] = question_dic[subject_type]
                    label[1 + pre_idnex + ans[0]: ans[1] + pre_idnex] = question_dic[subject_type] + 1
                input_ids.append(encode_dic['input_ids'])
                segment_ids.append(encode_dic['token_type_ids'])
                input_mask.append( encode_dic['attention_mask'])
                labels.append(label)
                context_lengths.append([ len(subject_question),len(context)])
                question_types.append(question_dic[subject_type])
                # features.append({
                #     'input_ids': encode_dic['input_ids'],
                #     'segment_ids': encode_dic['token_type_ids'],
                #     'input_mask': encode_dic['attention_mask'],
                #     'label': label,
                #     'context_lengths': [ len(subject_question),len(context)],
                #     # 'context': [context, subject_question],
                #     'question_type': question_dic[subject_type]
                # })
            else:
                relation = line['relation_type']
                try :

                    rel_question =line['subject_token'] + relation2question[relation]
                except Exception as e:
                    print(e,line)
                encode_dic = tokenizer.encode_plus(text=[char for char in context],
                                                   text_pair=[char for char in rel_question],
                                                   max_length=max_seq_length,
                                                   add_special_tokens=True, pad_to_max_length=True,truncation=True)
                label = np.zeros(max_seq_length, dtype=np.long)
                indexs = line['label']
                if indexs != []:
                    pre_idnex = 1
                    if type(indexs[0]) ==list:
                        for ans in indexs:
                            label[pre_idnex + ans[0]] = question_dic[line['object_type']]
                            label[1 + pre_idnex + ans[0]: ans[1] + pre_idnex] = question_dic[line['object_type']] + 1
                    else :
                        label[pre_idnex + indexs[0]] = question_dic[line['object_type']]
                        label[1 + pre_idnex + indexs[0]: indexs[1] + pre_idnex] = question_dic[line['object_type']] + 1

                input_ids.append(encode_dic['input_ids'])
                segment_ids.append(encode_dic['token_type_ids'])
                input_mask.append(encode_dic['attention_mask'])
                labels.append(label)
                context_lengths.append([len(rel_question), len(context)])
                question_types.append(question_dic[relation])
                #
                # features.append({
                #     'input_ids': encode_dic['input_ids'],
                #     'segment_ids': encode_dic['token_type_ids'],
                #     'input_mask': encode_dic['attention_mask'],
                #     'label': label,
                #     'context_lengths': [ len(rel_question),len(context)],
                #     'question_type': question_dic[relation]
              # })
    np.save(filename.split('.json')[0] + '_input_ids.npy', input_ids)
    np.save(filename.split('.json')[0] + '_segment_ids.npy', segment_ids)
    np.save(filename.split('.json')[0] + '_input_masks.npy', input_mask)
    np.save(filename.split('.json')[0] + '_labels.npy', labels)
    np.save(filename.split('.json')[0] + '_context_lengths.npy', context_lengths)
    np.save(filename.split('.json')[0] + '_question_types.npy', question_types)




def convert_to_Input(data_dir,tokenizer,max_length):
    print('processing train_qa.json')
    convert_to_input_feature(os.path.join(data_dir,'train_qa_modifiedLymph_artificial_question_neg_format.json'),tokenizer,max_length)
    # convert_to_input_feature(os.path.join(data_dir,'train_qa_disease_5.json'),tokenizer,max_length)
    # convert_to_input_feature(os.path.join(data_dir,'train_qa_5.json'),tokenizer,max_length)
    print('processing valid_qa.json')
    convert_to_input_feature(os.path.join(data_dir,'valid_qa_modifiedLymph_artificial_question_neg_format.json'),tokenizer,max_length)
    # convert_to_input_feature(os.path.join(data_dir,'valid_qa_disease_5.json'),tokenizer,max_length)
    # convert_to_input_feature(os.path.join(data_dir,'valid_qa_5.json'),tokenizer,max_length)
    print('processing test_qa.json')
    convert_to_input_feature(os.path.join(data_dir,'test_qa_modifiedLymph_artificial_question_neg_format.json'),tokenizer,max_length)
    # convert_to_input_feature(os.path.join(data_dir,'test_qa_disease_5.json'),tokenizer,max_length)
    # convert_to_input_feature(os.path.join(data_dir,'test_qa_5.json'),tokenizer,max_length)


def get_test_text(test_filename,max_length):
    with open(test_filename,'r',encoding='utf8') as f_source:
        with open('./test_data.txt','w',encoding='utf8') as f_out_txt:
            with open('./test_data_clean.json','w',encoding='utf8') as f_out_json:
                for line in f_source:
                    data_line = json.loads(line)
                    spo_list = data_line['spo_list']
                    context = clean_data(data_line['text'])
                    if len(context)<max_length-20:
                        f_out_txt.write(context)
                        f_out_txt.write('\n')
                        spo_lst_clean = []
                        for spo in spo_list:
                            subject_type = spo['subject_type']
                            object_type = spo['object_type']['@value']
                            subject = re.sub(r',', "，", spo['subject'])
                            subject = re.sub(r'[(]', '（', subject)
                            subject = re.sub(r'[)]', '）', subject)
                            object = re.sub(r',', "，", spo['object']['@value'])
                            object = re.sub(r'[(]', '（', object)
                            object = re.sub(r'[)]', '）', object)
                            spo_lst_clean.append({'subject_type':subject_type,
                                                  'subject':subject,
                                                  'object_type':object_type,
                                                  'object':object,
                                                  'relation':spo['predicate']
                                                  })
                        f_out_json.write(json.dumps({"text":context,
                                                     "spo_list":spo_lst_clean
                                                     },ensure_ascii=False))
                        f_out_json.write('\n')




if __name__ =='__main__':
    # data_dir =  r'G:\LungCancer\project\code\IE\Entity-Relation-As-Multi-Turn-QAV2\Entity-Relation-As-Multi-Turn-QA-master\data\data_modified\keywords_only'
    # print('dev')
    # convert_triple_to_qa_format_directly('./dev_data.json',mode='dev',num_neg_sample=8)
    # print('train')
    # convert_triple_to_qa_format_directly('./train_data.json',mode='train',num_neg_sample=8)
    # print('test')
    # print('train')
    # add_neg_samples(r'G:\LungCancer\project\code\IE\IE by multi-turn&stage QA\data\question\train_qa_modifiedLymph_artificial_question.json')
    # print('test')
    # add_neg_samples(r'G:\LungCancer\project\code\IE\IE by multi-turn&stage QA\data\question\test_qa_modifiedLymph_artificial_question.json')
    # print('dev')
    # add_neg_samples(r'G:\LungCancer\project\code\IE\IE by multi-turn&stage QA\data\question\valid_qa_modifiedLymph_artificial_question.json')
    tokenizer = BertTokenizer.from_pretrained(r'G:\DeepLearning\NLP\pre-trained-models\chinese-bert_chinese_wwm_pytorch')
    convert_to_Input(r'G:\LungCancer\project\code\IE\IE by multi-turn&stage QA\data\question',tokenizer,max_length= 168)
