import os
from os.path import exists, join
import json
from typing import *
from rouge import Rouge

rouge = Rouge()


def eval_rouge_bangla( dec: str, ref: str) -> Tuple[float, float, float]:
    """
    Use Rouge library to naively calculate rouge score for Bangla
    :param dec: Hypothesis/candidate summary
    :param ref: Ground Truth summary
    :return:
    """
    if dec == '' or ref == '':
        return 0.0, 0.0, 0.0
    scores = rouge.get_scores(dec, ref)
    return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']


def fast_rouge(dec : str, reference : str) -> float:
    """
    Calculate a naive rouge score
    :param dec: Hypothesis or candidate summary
    :param reference: Reference summary (ground truth)
    :return:
    """
    if dec == '' or reference == '':
        return 0.0
    scores = rouge.get_scores(dec, reference)
    return (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3




def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_data_path(mode, encoder):
    paths = {}
    if mode == 'train':
        paths['train'] = 'data/train_CNNDM_' + encoder + '.jsonl'
        paths['val']   = 'data/val_CNNDM_' + encoder + '.jsonl'
    else:
        paths['test']  = 'data/test_CNNDM_' + encoder + '.jsonl'
    return paths

def get_result_path(save_path, cur_model):
    result_path = join(save_path, '../result')
    if not exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, cur_model)
    if not exists(model_path):
        os.makedirs(model_path)
    dec_path = join(model_path, 'dec')
    ref_path = join(model_path, 'ref')
    os.makedirs(dec_path)
    os.makedirs(ref_path)
    return dec_path, ref_path

def merge_array_of_strings(str_of_arr: [str])->str:
    return " ".join(str_of_arr)


