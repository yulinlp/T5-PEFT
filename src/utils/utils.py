import json
from easydict import EasyDict
import numpy as np
from torch.utils.data import Dataset
import logging

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

def clean_data(dataset):
    '''
    清洗数据
    '''
    train_data = _clean_data(dataset, catagory='train')
    test_data = _clean_data(dataset, catagory='test')
    valid_data = _clean_data(dataset, catagory='validation')
    
    return train_data, test_data, valid_data

def _clean_data(dataset, catagory='train'):
    data = list(map(eval, dataset[catagory]['text']))
    data = [item['dialog'] for item in data]
    # 合并相邻的相同speaker的数据, 处理sys先开口的情况
    for i in range(len(data)):
        for j in range(len(data[i]) - 1, 0, -1):
            if data[i][j]['speaker'] == data[i][j - 1]['speaker']:
                data[i][j - 1]['text'] += data[i][j]['text']
                data[i].pop(j)
    data = [[(dialog[idx - 1]['text'], dialog[idx]['text']) for idx in range(1, len(dialog), 2)] if dialog[0]['speaker']=='usr' else [(dialog[idx - 1]['text'], dialog[idx]['text']) for idx in range(2, len(dialog), 2)] for dialog in data]
    # 构造带有历史的对话数据
    new_data = []
    for idx, item in enumerate(data):
        for i in range(len(item)):
            new_data.append({
                'input': build_template_default(query=item[i][0], prefix=False, history=item[:i] if i > 0 else None),
                'target': item[i][1]
            })
    
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

def postprocess_text(preds, labels):
    '''
    文本后处理
    '''
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

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
