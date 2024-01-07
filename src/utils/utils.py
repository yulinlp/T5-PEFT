import json
from easydict import EasyDict
import numpy as np
from torch.utils.data import Dataset
import logging
import torch
import random

from tqdm import tqdm

def read_json(path, n_rows=None):
    '''
    读json文件
    '''
    with open(path, 'r') as f:
        data = json.load(f)
    return data if n_rows is None else data[:n_rows]

def get_config(json_path):
    '''
    json -> Config
    '''
    config = EasyDict(read_json(json_path))
    return config

def clean_data(dataset, is_official=True):
    '''
    清洗数据
    '''
    if is_official:
        train_data = _clean_data_official(dataset, catagory='train')
        test_data = _clean_data_official(dataset, catagory='test')
        valid_data = _clean_data_official(dataset, catagory='validation')
    else:
        train_data = _clean_data(dataset, catagory='train')
        test_data = _clean_data(dataset, catagory='test')
        valid_data = _clean_data(dataset, catagory='validation')
    
    return train_data, test_data, valid_data

def _clean_data(dataset, catagory='train'):
    data = list(map(eval, dataset[catagory]['text']))
    data = [item['dialog'] for item in data]
    # 合并相邻的相同speaker的数据
    for i in range(len(data)):
        for j in range(len(data[i]) - 1, 0, -1):
            if data[i][j]['speaker'] == data[i][j - 1]['speaker']:
                data[i][j - 1]['text'] += " " + data[i][j]['text']
                data[i].pop(j)
    # 处理sys先开口的情况：直接丢弃sys第一句话，始终保持usr先开口
    data = [[(dialog[idx - 1]['text'], dialog[idx]['text']) for idx in range(1, len(dialog), 2)] if dialog[0]['speaker']=='usr' else [(dialog[idx - 1]['text'], dialog[idx]['text']) for idx in range(2, len(dialog), 2)] for dialog in data]

    return data

def _clean_data_official(dataset, catagory='train'):
    '''
    将每个对话分成 5 个话语的对话片段, 其中包含一位支持者的回应和前4句话。
    '''
    data = list(map(eval, dataset[catagory]['text']))
    data = [item['dialog'] for item in data]
    new_data = []
    for dialog in data:
        for i in range(5, len(dialog)):
            if dialog[i - 1]['speaker'] == 'sys':
                new_data.append(dialog[i-5:i])
    
    return new_data
    
def merge_history_official(data, tokenizer=None):
    # 构造带有历史的对话数据
    if tokenizer.additional_special_tokens is not None:
        new_data = []
        for dialog in data:
            new_dialog = []
            for idx, utterance in enumerate(dialog):
                if utterance['speaker'] == 'sys':
                    text = tokenizer.additional_special_tokens[1] + utterance['text']
                else:
                    text = tokenizer.additional_special_tokens[0] + utterance['text']
                new_dialog.append(text)
            new_data.append(new_dialog)
    
    ret_data = []
    for dialog in new_data:
        ret_data.append({
            'input': ''.join(dialog[:-1]),
            'target': dialog[-1] + tokenizer.eos_token if tokenizer.additional_special_tokens is not None else dialog[-1]
        })
    return ret_data

def merge_history(data, tokenizer = None):
    # 构造带有历史的对话数据
    if tokenizer.additional_special_tokens is not None:
        # 对每轮每人的发言加入eos, 起始加入bos
        for idx, item in enumerate(data):
            for i in range(len(item)):
                if i == 0:
                    item[i] = (tokenizer.bos_token + item[i][0], tokenizer.additional_special_tokens[1] + item[i][1])
                    # item[i] = (tokenizer.additional_special_tokens[0] + item[i][0], tokenizer.additional_special_tokens[1] + item[i][1])
                else:
                    item[i] = (tokenizer.additional_special_tokens[0] + item[i][0], tokenizer.additional_special_tokens[1] + item[i][1])
    
    new_data = []
    for idx, item in enumerate(data):
        for i in range(len(item)):
            new_data.append({
                'input': build_template_default(query=item[i][0], prefix=False, history=item[:i] if i > 0 else None),
                'target': item[i][1] + tokenizer.eos_token if tokenizer.additional_special_tokens is not None else item[i][1]
            })
    # exit()
    return new_data

def build_template_default(query, prefix=False, history=None):
    '''
    构造prompt模板
    '''
    prompt = ''
    if prefix:
        if history is not None:
            for q,a in history:
                if q == '':
                    prompt += "Assistant: {}\n".format(a)
                else:
                    prompt += "User: {}\nAssistant: {}\n".format(q,a)
        if query != '':
            prompt += "User: {}\nAssistant: ".format(query)
        else:
            prompt += "Assistant: "
    else:
        if history is not None:
            for q,a in history:
                if q == '':
                    prompt += "{}\n".format(a)
                else:
                    prompt += "{}\n{}\n".format(q,a)
        if query != '':
            prompt += "{}\n".format(query)
    return prompt

def count_len(dataset: Dataset):
    '''
    统计dataset中的数据长度分布
    '''
    lens = []
    for idx in range(len(dataset)):
        lens.append(len(dataset[idx]))
    import matplotlib.pyplot as plt
    
    # 绘制直方图
    plt.scatter(range(len(lens)), lens)
    plt.show()
    print("success")

def create_logger(logger_file_name):
    """
    创建logger
    """
    logger = logging.getLogger()         # 设定日志对象
    logger.setLevel(logging.INFO)        # 设定日志等级

    file_handler = logging.FileHandler(logger_file_name)   # 文件输出
    console_handler = logging.StreamHandler()              # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)       # 设置文件输出格式
    console_handler.setFormatter(formatter)    # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def find_last_index(arr, target):
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] == target:
            return i
    return -1