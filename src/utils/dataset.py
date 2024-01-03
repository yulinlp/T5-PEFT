from typing import *
from torch.utils.data import Dataset


class T5Dataset(Dataset):
    def __init__(self, data: List, Config, tokenizer) -> None:
        self.max_input_length = Config.max_input_length
        self.max_output_length = Config.max_output_length
        self.data = self.__preprocess(data, tokenizer)
        # print(type(self.data))
        # print(self.data[0])
    
    def __getitem__(self, index) -> Any:
        return {"labels": self.data['labels'][index], "input_ids": self.data["input_ids"][index], "attention_mask": self.data['attention_mask'][index]}
        # return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data["input_ids"])
    
    def __preprocess(self, data, tokenizer, padding="max_length"):
        '''
        最后对数据的预处理
        '''
        # tokenize inputs
        inputs = [item['input'] for item in data]
        inputs = tokenizer(inputs, max_length=self.max_input_length, padding=padding, truncation=True)
        # Tokenize targets with the `target` keyword argument
        targets = [item['target'] for item in data]
        targets = tokenizer(targets, max_length=self.max_output_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            targets["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in targets["input_ids"]
            ]

        inputs['labels'] = targets["input_ids"]
        return inputs