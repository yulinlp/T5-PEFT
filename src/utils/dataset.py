from typing import *
from typing import Any, Dict, List, Union
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch
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
        self.model_name = ModelConfig.model_name.lower()
        self.data = self.__preprocess(data, tokenizer, Config.padding, Config.truncation, Config.data_official)
    
    def __getitem__(self, index) -> Any:
        return {"input_ids": self.data["input_ids"][index], "labels": self.data["labels"][index], "attention_mask": self.data['attention_mask'][index]}
        # return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data["input_ids"])
    
    def __preprocess(self, data, tokenizer, padding="max_length", truncation="from_end", data_official=False):
        if data_official:
            data = merge_history_official(data, tokenizer)
        else:
            data = merge_history(data, tokenizer)
        # print(data[0])
        # exit()
        # tokenize inputs
        inputs = [item['input'] + item['target'] for item in data]
        targets = [item['target'] for item in data]
        if truncation == "from_end":
            inputs = tokenizer(inputs, max_length=self.max_input_length, padding=padding, truncation=False)
            new_input_ids = [input[-self.max_input_length:] if len(input) > self.max_input_length else input for input in inputs['input_ids']]
            inputs['input_ids'] = new_input_ids

            # new_attention_mask = [mask[-self.max_input_length:] if len(mask) > self.max_input_length else mask for mask in inputs['attention_mask']]
            # inputs['attention_mask'] = new_attention_mask
            
            targets = tokenizer(targets, max_length=self.max_output_length, padding=False, truncation=True)['input_ids']
            # exit()
            inputs['labels'] = []
            for i in range(len(targets)):
                label = []
                target_idx = find_sublist_index(inputs['input_ids'][i], targets[i])
                for j in range(len(inputs['input_ids'][i])):
                    if j < target_idx or j >= (target_idx + len(targets[i])):
                        label.append(-100)
                    else:
                        label.append(inputs['input_ids'][i][j])
                inputs['labels'].append(label)
        else:
            inputs = tokenizer(inputs, max_length=self.max_input_length, padding=padding, truncation=True)
        return inputs


class GPT_data_collator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, pad_to_multiple_of=None, mlm=False, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of, mlm=mlm, mlm_probability=mlm_probability)
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["labels"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch
