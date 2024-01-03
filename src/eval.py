import torch
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import warnings
import os
from datasets import load_dataset
from random import randrange
from utils.dataset import T5Dataset
from utils.utils import *
import evaluate

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore")

EvalConfig = get_config('config/eval.json')
ModelConfig = get_config('config/model.json')

print(ModelConfig)
print(EvalConfig)

if EvalConfig.is_PEFT:
    # Load peft config for pre-trained checkpoint etc.
    peft_model_id = EvalConfig.PEFT_model_dir
    config = PeftConfig.from_pretrained(peft_model_id)
    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
else:
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(ModelConfig.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

model.eval()
print("Model loaded") 

raw_data = load_dataset("./dataset/esconv")
print("Successfully loaded dataset")

_, test_data, _ = clean_data(raw_data)
test_set = T5Dataset(test_data, EvalConfig, tokenizer)
# sample = test_data[randrange(len(test_data))]
# input_ids = tokenizer(sample["input"], return_tensors="pt", truncation=True).input_ids.cuda()
# outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9)

# print(f"inputs: {sample['input']}\n{'---'* 20}")
# print(f"preds: \n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}") 
# print(f"GT: {sample['target']}")

def evaluate_peft_model(sample):
    # generate summary
    outputs = model.generate(input_ids=torch.tensor(sample["input_ids"]).unsqueeze(dim=0).cuda(), do_sample=True, top_p=0.9, max_new_tokens=EvalConfig.max_output_length, min_new_tokens=1)
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=False)
    # decode eval sample
    # Replace -100 in the labels as we can't decode them.
    labels = [0 if val == -100 else val for val in sample['labels']]
    labels = tokenizer.decode(labels, skip_special_tokens=False)

    return prediction, labels


predictions, references = [], []
for sample in tqdm(test_set, desc="Generating"):
    p,l = evaluate_peft_model(sample)
    predictions.append(p)
    references.append(l)

# for p, r in zip(predictions, references):
#     if p == '':
#         # print(p)
#         print(r)
#         print('#'*30) 

result = {}
for metric_name in tqdm(EvalConfig.metrics, desc="Evaluating"):
    print(f"Loading {metric_name}...")
    metric = evaluate.load(path=f'./cached_metrics/{metric_name}')
    print(f"Computing {metric_name}...")
    
    if metric_name == "bleu":
        partial_result = metric.compute(predictions=predictions, references=references)
        result["bleu-2"] = partial_result["precisions"][1]
    elif metric_name == "rouge":
        partial_result = metric.compute(predictions=predictions, references=references)
        result["rouge-L"] = partial_result["rougeL"]
    elif metric_name == "perplexity":
        partial_result = metric.compute(model_id='gpt2',
                            add_start_token=False,
                            predictions=predictions)
        
        result["perplexity"] = partial_result["mean_perplexity"]

print(f"bleu-2: {result['bleu-2']}")
print(f"rouge-L: {result['rouge-L']}")
print(f"perplexity: {result['perplexity']}")