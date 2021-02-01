#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

# Author: Huanyao Zhang
# Last update: 2020.10.29
# First create: 2019.03.23

import json
import os
import sys 
import math 
import numpy as np 
import logging
import pandas as pd
from data.data_utils import entity2id, id2entity
import torch

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
# print(root_path)
if root_path not in sys.path:
    sys.path.append(root_path)

log = logging.getLogger(__name__)

def rectify_indices(context,pred_indices,gold_indeices_len):
     
    if pred_indices == [] :
        return pred_indices
    if gold_indeices_len == 1 and len(pred_indices) > 1:
        pred_indices = sorted(pred_indices,key=lambda x:x[0])
        pred_indices = [[pred_indices[0][0],pred_indices[-1][1]]]
    rectified_indices = []
    for indices in pred_indices :
        text = context[indices[0]:indices[1]]
        if "（" in text:
            indices = [indices[0],indices[0] + text.index("（")]
           # rectified_indices.append(indices)
        elif "。" in text:
            indices = [indices[0],indices[0] + text.index("。")]
        
        rectified_indices.append(indices) 
    return  rectified_indices


def cal_f1_score(precision, rec):
    tmp = 2 * precision * rec / (precision + rec)
    return round(tmp, 4)



def cal_triple_res(gold_filename,predict_filename):
    data_gold,data_predict = [],[]
    with open(gold_filename,'r',encoding='utf8') as f:
        for line in f:
            data_gold.append(json.loads(line))
    with open(predict_filename,'r',encoding='utf8') as f:
        for line in f:
            data_predict.append(json.loads(line))
    len_gold = len(data_gold)
    len_predict = len(data_predict)
    assert len_gold == len_predict

    match_count = 0
    for ix in range(len_gold):
        spo_gold_list = data_gold[ix]['spo_list']
        spo_predict_list = data_predict[ix]['spo_list']
        match_count += 1
        for spo_predict in spo_predict_list:

            if spo_predict not in spo_gold_list:
                match_count-=1
                break

    print('precision = {}'.format(match_count/len_gold))

    return (match_count,match_count/len_gold)


def get_indices_from_lable(label, index_number):
    indices = []
    start_index = 0
    label_len = len(label)
    while start_index < label_len:
        # print('start_index = ',start_index ,'label = ',label[start_index])

        if label[start_index] == index_number:
            # print('start_index = ',start_index ,'label = ',label[start_index])

            end_index = start_index + 1
            while end_index < label_len and label[end_index] == index_number + 1:
                end_index += 1
            indices.append([start_index, end_index])
            start_index = end_index - 1
        start_index += 1
    return indices


def get_index_number(gold_label):
    index_number = []
    # flag = 0
    for gold in gold_label:
        flag = 0
        for num in gold:
            if num != 0:
                index_number.append(num)
                flag = 1
                break
        if not flag:
            index_number.append(0)

    return index_number


def get_blurred_match_count(gold_indices, pred_indices):
    for pred_indice in pred_indices:
        for gold_indice in gold_indices:
            if not (pred_indice[0] > gold_indice[1] and pred_indice[1] < gold_indice[0]):
                return 1
    return 0


def compute_performance(pred_label, gold_label, lengths, question_types, config, sign='dev', context=None,  threadhold=0.5):
    # start_label = split_index(label_list)
    assert len(pred_label) == len(gold_label)
    performance_dic = {}
    res_lst = []
    for key in entity2id.keys():
        if '-' not in key and key not in ['<unk>', 'O', '<START>', '<STOP>']:
            performance_dic[key] = {
                'tp_precise': 0,
                'num_pre': 0,
                'num_gold': 0,
                'tp_blurred': 0,

            }
    precision_lst_precise = []
    recall_lst_precise = []
    f1_lst_precise = []
    precision_lst_blurred = []
    recall_lst_blurred = []
    f1_lst_blurred = []
    res_lst = []
    key_lst = []
    match_count = 0
    if sign == 'test' or sign =="train":
        output_file = os.path.join(config.output_dir, '{}_res.json'.format(sign))


    if type(gold_label) == tuple and type(pred_label) == tuple:
        gold_starts = gold_label[0]
        gold_ends = gold_label[1]
        pre_starts = pred_label[0]
        pre_ends = pred_label[1]
        print(gold_starts[0])
        for ix in range(len(pre_starts)):
            gold_start_inidices = [jx for jx in range(len(gold_starts[ix])) if gold_starts[ix][jx] > 0]
            gold_end_inidices = [jx for jx in range(len(gold_ends[ix])) if gold_ends[ix][jx] > 0]
            if gold_start_inidices != [] and len(gold_start_inidices) == len(gold_end_inidices):
                pre_start_inidices = [jx for jx in range(len(pre_starts[ix])) if pre_starts[ix][jx] > 0]
                pre_end_inidices = [jx for jx in range(len(pre_ends[ix])) if pre_ends[ix][jx] > 0]
                print('gold_start_inidices = {}, gold_end_inidices = {}'.format(gold_start_inidices,
                                                                                gold_end_inidices))
                print('pre_start_inidices = {}, pre_end_inidices = {}'.format(pre_start_inidices, pre_end_inidices))

                gold_indices = [[gold_start_inidices[ix], gold_end_inidices[ix]] for ix in
                                range(len(gold_start_inidices))]
                tp = 0

                if len(pre_start_inidices) == len(pre_end_inidices):

                    # match_count = 0
                    for ix in range(len(pre_start_inidices)):
                        if [pre_start_inidices[ix], pre_end_inidices[ix]] in gold_indices:
                            tp += 1
                            match_count += 1
                performance_dic[id2entity[question_types[ix].item()]]['tp'] += tp
                performance_dic[id2entity[question_types[ix].item()]]['num_pre'] += 1 if len(
                    pre_start_inidices) != 0 else 0
                performance_dic[id2entity[question_types[ix].item()]]['num_gold'] += 1 if len(
                    gold_start_inidices) != 0 else 0

                if sign == 'test':
                    res_lst.append({
                        'context': context[ix][0],
                        'question': context[ix][1],
                        'answer_gold': [context[ix][0][gold_start_inidices[ix]:gold_end_inidices[1] + 1] for ix in
                                        range(len(gold_start_inidices))],
                        'answer_pred': [context[ix][0][pre_start_inidices[ix]:pre_end_inidices[1] + 1] for ix in
                                        range(len(pre_start_inidices))],
                        'question_type': id2entity[question_types[ix].item()]
                    })
        for key in performance_dic.keys():
            performance_dic[key]['precision'] = performance_dic[key]['tp'] / performance_dic[key]['num_pre'] if \
            performance_dic[key]['num_pre'] != 0 else 0
            performance_dic[key]['recall'] = performance_dic[key]['tp'] / performance_dic[key]['num_gold'] if \
            performance_dic[key]['num_gold'] != 0 else 0
            performance_dic[key]['f1'] = 2 * performance_dic[key]['recall'] * performance_dic[key]['precision'] / (
                    performance_dic[key]['recall'] + performance_dic[key]['precision']) if performance_dic[key][
                                                                                               'recall'] != 0 and \
                                                                                           performance_dic[key][
                                                                                               'precision'] != 0 else 0
            f1_lst.append(performance_dic[key]['f1'])
            recall_lst.append(performance_dic[key]['recall'])
            precision_lst.append(performance_dic[key]['precision'])
            print('key :{} ,value :{}'.format(key, performance_dic[key]))
        if sign == 'test':
            res_lst.append(performance_dic)
            with open(output_file, 'w', encoding='utf8') as f:
                json.dump(res_lst, f, indent=4, ensure_ascii=False)
    else:

        index_numbers = get_index_number(gold_label)
        assert len(pred_label) == len(gold_label) == len(index_numbers) == len(question_types)
        # index_numbers = get_index_number(gold_label)
        match_count = 0

        for ix in range(len(index_numbers)):
        # for postive sample
            tp_precice = 0
            tp_blurred = 0
            match_flag = 1
            if index_numbers[ix] != 0:

                pred_indices = get_indices_from_lable(pred_label[ix], index_numbers[ix])

                gold_indices = get_indices_from_lable(gold_label[ix], index_numbers[ix])
   

                if len(pred_indices) != 0 and len(gold_indices) != 0:

                    for pre_indice in pred_indices:
                        if pre_indice  in  gold_indices:
                            tp_precice += 1
                            match_count += 1

                    tp_blurred = get_blurred_match_count(gold_indices, pred_indices)

                if sign == 'test' or  sign ==  "train":
                    res_lst.append({
                        'answer_gold': [[indice[0],indice[1]] for indice in gold_indices],
                        'answer_pred': [[indice[0],indice[1]] for indice in pred_indices],
                        'question_type': id2entity[question_types[ix]]
                         })
            # for negative sample
            else:
                gold_indices = [0]
                if len(set(pred_label[ix])) == 1 :
                    tp_precice = 1
                    pred_indices = []
                    
                else:
                    tp_precice = 0
                    index_numbers_pre = get_index_number([pred_label[ix]])
                    pred_indices = [ get_indices_from_lable(pred_label[ix], index_number) for index_number in index_numbers_pre]
                tp_blurred = tp_precice

                if sign == 'test' or  sign ==  "train":

                    res_lst.append({
                        'answer_gold': gold_indices,
                        'answer_pred': pred_indices ,
                        'question_type': id2entity[question_types[ix]]
                    })
            performance_dic[id2entity[question_types[ix]]]['tp_precise'] += tp_precice
            performance_dic[id2entity[question_types[ix]]]['tp_blurred'] += tp_blurred

            performance_dic[id2entity[question_types[ix]]]['num_pre'] += len(pred_indices) if len(pred_indices) !=0 else 1
            performance_dic[id2entity[question_types[ix]]]['num_gold'] += len(gold_indices) if len(gold_indices) !=0 else 1



        for key in performance_dic.keys():
            performance_dic[key]['precision_precise'] = performance_dic[key]['tp_precise'] / performance_dic[key][
                'num_pre'] if performance_dic[key]['num_pre'] != 0 else 0
            performance_dic[key]['recall_precise'] = performance_dic[key]['tp_precise'] / performance_dic[key][
                'num_gold'] if performance_dic[key]['num_gold'] != 0 else 0
            performance_dic[key]['f1_precise'] = 2 * performance_dic[key]['recall_precise'] * performance_dic[key][
                'precision_precise'] / (performance_dic[key]['recall_precise'] + performance_dic[key][
                'precision_precise']) if performance_dic[key]['recall_precise'] != 0 and performance_dic[key][
                'precision_precise'] != 0 else 0
            performance_dic[key]['precision_blurred'] = performance_dic[key]['tp_blurred'] / performance_dic[key][
                'num_pre'] if performance_dic[key]['num_pre'] != 0 else 0
            performance_dic[key]['recall_blurred'] = performance_dic[key]['tp_blurred'] / performance_dic[key][
                'num_gold'] if performance_dic[key]['num_gold'] != 0 else 0
            performance_dic[key]['f1_blurred'] = 2 * performance_dic[key]['recall_blurred'] * performance_dic[key][
                'precision_blurred'] / (performance_dic[key]['recall_blurred'] + performance_dic[key][
                'precision_blurred']) if performance_dic[key]['recall_blurred'] != 0 and performance_dic[key][
                'precision_blurred'] != 0 else 0

            f1_lst_precise.append(performance_dic[key]['f1_precise'])
            recall_lst_precise.append(performance_dic[key]['recall_precise'])
            precision_lst_precise.append(performance_dic[key]['precision_precise'])
            f1_lst_blurred.append(performance_dic[key]['f1_blurred'])
            recall_lst_blurred.append(performance_dic[key]['recall_blurred'])
            precision_lst_blurred.append(performance_dic[key]['precision_blurred'])
            key_lst.append(key)
            df = pd.DataFrame(np.array(
                [key_lst, precision_lst_blurred, recall_lst_blurred, f1_lst_blurred, precision_lst_precise,
                 recall_lst_precise, f1_lst_precise]).T,
                              columns=['queation_type', 'precision_blurred', 'recall_blurred', 'f1_blurred',
                                       'precision_precise', 'recall_precise', 'f1_precise'])
         #   print(df)

        if sign == 'test' or sign == "train":

            df.to_excel(config.output_dir + '/{}_res.xlsx'.format(sign), index=None, encoding='utf8')
            res_lst.append(performance_dic)
            with open(output_file, 'w', encoding='utf8') as f:
                json.dump(res_lst, f, indent=4, ensure_ascii=False)

    acc_score = match_count / len(gold_label)
    f1_score = round(sum(f1_lst_precise) / len(f1_lst_precise), 4) if f1_lst_precise != [] else 0
    recall_score = round(sum(recall_lst_precise) / len(recall_lst_precise), 4) if recall_lst_precise != [] else 0
    precision_score = round(sum(precision_lst_precise) / len(precision_lst_precise),
                            4) if precision_lst_precise != [] else 0

    return acc_score, precision_score, recall_score, f1_score






def eval_checkpoint(model_object, eval_dataloader, config, \
                    device, n_gpu, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model_object.eval()

    # idx2label = {i: label for i, label in enumerate(label_list)}

    eval_loss = 0
    pred_lst = []
    start_lst = []
    end_lst = []
    label_start_lst = []
    label_end_lst = []
    gold_lst = []
    length_lst = []
    eval_steps = 0
    # context_lst = []
    question_type_lst = []
    # if eval_sign == 'test':
    if config.start_end:
        for input_ids, input_mask, segment_ids, label_start, label_end, context_lengths, context, question_type in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_start = label_start.to(device)
            label_end = label_end.to(device)
            context_lengths = context_lengths.to(device)
            question_type = question_type.to(device)
            # context = [[context[0][ix], context[1][ix]] for ix in range(len(context[0]))]
            # print("context :{}".format(context))
            # logger.info("input mask: {}".format(input_mask))

            # print('in {:s} line{}, shape of context_lengths = {},shape of input_ids = {}'.format(__file__,sys._getframe().f_lineno,context_lengths.shape,input_ids.shape))
            with torch.no_grad():
                tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, (label_start, label_end),
                                             context_lengths)
                start_socre, end_score = model_object(input_ids, segment_ids, input_mask, labels=None,
                                                      context_lengths=context_lengths)

            # logits = torch.argmax(logits,dim=-1)
            length_lst += context_lengths.to("cpu").numpy().tolist()

            # logits = np.argmax(logits, axis=-1)
            label_start = label_start.to("cpu").numpy().tolist()
            label_end = label_end.to('cpu').numpy().tolist()
            start_socre = start_socre.detach().numpy()
            end_socre = end_score.detach().numpy()

            eval_loss += tmp_eval_loss.mean().item()

            # mask_lst += input_mask

            label_start = [label_start[ix][2 + length_lst[ix][0]: 2 + length_lst[ix][0] + length_lst[ix][1]] for ix in
                           range(len(label_start))]
            label_end = [label_end[ix][2 + length_lst[ix][0]: 2 + length_lst[ix][0] + length_lst[ix][1]] for ix in
                         range(len(label_end))]
            start_socre = [start_socre[ix][2 + length_lst[ix][0]: 2 + length_lst[ix][0] + length_lst[ix][1]] for ix in
                           range(len(start_socre))]
            end_score = [label_end[ix][2 + length_lst[ix][0]: 2 + length_lst[ix][0] + length_lst[ix][1]] for ix in
                         range(len(end_socre))]

            start_lst += start_socre
            end_lst += end_score
            label_start_lst += label_start
            label_end_lst += label_end

            question_type_lst += question_type
            # print('in {:s} line{}, context_lengths = {}'.format(__file__,sys._getframe().f_lineno,length_lst))
            eval_steps += 1
        pred_lst = (start_lst, end_lst)
        gold_lst = (label_start_lst, label_end_lst)
        eval_accuracy, eval_precision, eval_recall, eval_f1 = compute_performance(pred_lst, gold_lst, length_lst,
                                                                                  question_type_lst, config,
                                                                                   sign=eval_sign  )

        average_loss = round(eval_loss / eval_steps, 4)
        eval_f1 = round(eval_f1, 4)
        eval_precision = round(eval_precision, 4)
        eval_recall = round(eval_recall, 4)
        eval_accuracy = round(eval_accuracy, 4)

    else:
        for input_ids, input_mask, segment_ids, label_ids, context_lengths, question_type in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            context_lengths = context_lengths.to(device)
            question_type = question_type.to(device)
            # context = [[context[0][ix], context[1][ix]] for ix in range(len(context[0]))]
            # print("context :{}".format(context))
            # logger.info("input mask: {}".format(input_mask))

            # print('in {:s} line{}, shape of context_lengths = {},shape of input_ids = {}'.format(__file__,sys._getframe().f_lineno,context_lengths.shape,input_ids.shape))
            with torch.no_grad():
                tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, label_ids, context_lengths)
                logits = model_object(input_ids, segment_ids, input_mask, labels=None, context_lengths=context_lengths)

            # logits = torch.argmax(logits,dim=-1)
            context_lengths = context_lengths.to("cpu").numpy().tolist()
            length_lst += context_lengths

            # logits = np.argmax(logits, axis=-1)
            label_ids = label_ids.to("cpu").numpy()
            logits_lst = []
            # input_mask = input_mask.to("cpu").numpy()
            # logger.info( "in run _qa  :  length.shape = {} , logits[0] = {}".format(logits.shape,logits[0]))
            if config.use_crf:
                for ix in range(len(context_lengths)):
                    if config.question_first:
                        pre_index  = 2 + context_lengths[ix][0]
                    else:
                        pre_index = 0
                    log = logits[ix][pre_index:pre_index + context_lengths[ix][1]]
                    # print('log.shape  = {}, log = {}'.format(log.shape,log))
                    logits_lst.append(log)
                # print('in run_qa ,line {}, logits[:3] is {}'.format(sys._getframe().f_lineno,logits[:3]))
            # logits =torch.argmax(logits, dim=-1)
            else:
                logits = logits.squeeze(-1)

                for ix in range(len(context_lengths)):
                    log = logits[ix][2 + context_lengths[ix][0]: 2 + context_lengths[ix][0] + context_lengths[ix][1]]
                    # print('log.shape  = {}, log = {}'.format(log.shape,log))
                    logits_lst.append(log.argmax(-1).detach().cpu().numpy().tolist())
                # reshape_lst = label_ids.shape
                # logits = np.reshape(logits, (reshape_lst[0], reshape_lst[1], -1))
                # logits = logits.detach().cpu().numpy()
                # logits = logits.tolist()

                # logits = [logits[ix][2 + length_lst[ix][0]: 2 + length_lst[ix][0] + length_lst[ix][1] ] for ix in range(len(logits))]

            label_ids = label_ids.tolist()

            eval_loss += tmp_eval_loss.mean().item()
            if config.question_first:
                label_ids = [label_ids[ix][2 + context_lengths[ix][0]: 2 + context_lengths[ix][0] + context_lengths[ix][1]]
                         for ix in range(len(context_lengths))]
            else:
                label_ids = [
                    label_ids[ix][1: 1 + context_lengths[ix][1]] for ix in range(len(context_lengths))]

            pred_lst += logits_lst
            gold_lst += label_ids
            # context_lst += context
            question_type_lst += question_type.tolist()
            # print('in {:s} line{}, context_lengths = {}'.format(__file__,sys._getframe().f_lineno,length_lst))
            eval_steps += 1

        eval_accuracy, eval_precision, eval_recall, eval_f1 = compute_performance(pred_lst, gold_lst, length_lst,
                                                                                  question_type_lst, config,
                                                                                  context=None, sign=eval_sign)

        average_loss = round(eval_loss / eval_steps, 4)
        eval_f1 = round(eval_f1, 4)
        eval_precision = round(eval_precision, 4)
        eval_recall = round(eval_recall, 4)
        eval_accuracy = round(eval_accuracy, 4)



    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1


if __name__ == "__main__":
    # model_pred = [0, 1, 2, 3, 0, 1, 2, 3, 0, 4]
    #
    # # print(entities)
    #
    # label_list = np.array(["O", "B-NS", "M-NS", "E-NS","B-PER", "M-PER", "E-PER"])
    # start_label = split_index(label_list)
    # print(start_label)
    #
    # entities = extract_entities(model_pred, start_label = start_label)
    # for indexs,entity_indx in entities.items():
    #
    #     print('idnex of model predict =' ,indexs,'tag of idnex = ',label_list[entity_indx])
    gold_label = [[0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2], [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0]]
    pred_label = [[0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 0]]
    lengths = [len(label) for label in gold_label]
    compute_performance(pred_label, gold_label, lengths, dims=1)
