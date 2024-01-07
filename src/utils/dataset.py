from typing import *
from torch.utils.data import Dataset
from utils.utils import *

class T5Dataset(Dataset):
    def __init__(self, data: List, Config, ModelConfig, tokenizer) -> None:
        self.max_input_length = Config.max_input_length
        self.max_output_length = Config.max_output_length
        self.data = self.__preprocess(data, tokenizer, Config.padding, Config.truncation, Config.data_official)
        # print(type(self.data))
        # print(self.data[0])
    
    def __getitem__(self, index) -> Any:
        return {"labels": self.data['labels'][index], "input_ids": self.data["input_ids"][index], "attention_mask": self.data['attention_mask'][index]}
        # return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data["input_ids"])
    
    def __preprocess(self, data, tokenizer, padding="max_length", truncation="from_end", data_official=False):
        '''
        最后对数据的预处理
        '''
        if data_official:
            data = merge_history_official(data, tokenizer)
        else:
            data = merge_history(data, tokenizer)
            
        inputs = [item['input'] + tokenizer.additional_special_tokens[1] for item in data]
        targets = [item['target'].replace(tokenizer.eos_token, '').replace(tokenizer.additional_special_tokens[1], '') for item in data]
        # print(inputs[0])
        # print(targets[0])
        # exit()
        if truncation == "from_end":
            inputs = tokenizer(inputs, max_length=self.max_input_length, padding=padding, truncation=False)
            new_input_ids = [input[-self.max_input_length:] if len(input) > self.max_input_length else input for input in inputs['input_ids']]
            inputs['input_ids'] = new_input_ids

            new_attention_mask = [mask[-self.max_input_length:] if len(mask) > self.max_input_length else mask for mask in inputs['attention_mask']]
            inputs['attention_mask'] = new_attention_mask
        else:
            inputs = tokenizer(inputs, max_length=self.max_input_length, padding=padding, truncation=True)
        
        # Tokenize targets with the `target` keyword argument
        targets = tokenizer(targets, max_length=self.max_output_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            targets["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in targets["input_ids"]
            ]
            
        inputs['labels'] = targets["input_ids"]
        return inputs


class GPT2Dataset(Dataset):
    def __init__(self, data: List, Config, ModelConfig, tokenizer) -> None:
        self.max_input_length = Config.max_input_length
        self.max_output_length = Config.max_output_length
        self.model_name = ModelConfig.model_name
        self.data = self.__preprocess(data, tokenizer, Config.padding, Config.truncation, Config.data_official)
    
    def __getitem__(self, index) -> Any:
        return {"input_ids": self.data["input_ids"][index], "attention_mask": self.data['attention_mask'][index]}
        # return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data["input_ids"])
    
    def __preprocess(self, data, tokenizer, padding="max_length", truncation="from_end", data_official=False):
        if data_official:
            data = merge_history_official(data, tokenizer)
        else:
            data = merge_history(data, tokenizer)
        # tokenize inputs
        inputs = [item['input'] + item['target'] for item in data]
        if truncation == "from_end":
            inputs = tokenizer(inputs, max_length=self.max_input_length, padding=padding, truncation=False)
            new_input_ids = [input[-self.max_input_length:] if len(input) > self.max_input_length else input for input in inputs['input_ids']]
            inputs['input_ids'] = new_input_ids

            new_attention_mask = [mask[-self.max_input_length:] if len(mask) > self.max_input_length else mask for mask in inputs['attention_mask']]
            inputs['attention_mask'] = new_attention_mask
        else:
            inputs = tokenizer(inputs, max_length=self.max_input_length, padding=padding, truncation=True)
            
        return inputs