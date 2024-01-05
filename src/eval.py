import math
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
GenerateConfig = EasyDict(EvalConfig.GenerateConfig)
seed_everything(EvalConfig.seed)

print(ModelConfig)
print(EvalConfig)

metrics = {}
for metric_name in EvalConfig.metrics:
    metrics[metric_name] = evaluate.load(metric_name)
    print(f"Loaded {metric_name}...")

if EvalConfig.is_PEFT:
    # Load peft config for pre-trained checkpoint etc.
    peft_model_id = EvalConfig.PEFT_model_dir
    config = PeftConfig.from_pretrained(peft_model_id)
    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
    print("PEFT Model loaded") 
else:
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(ModelConfig.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("BASE Model loaded") 

model.eval()

raw_data = load_dataset("./dataset/esconv")
print("Successfully loaded dataset")

train_data, test_data, valid_data = clean_data(raw_data)
train_set = T5Dataset(train_data, EvalConfig, tokenizer)
test_set = T5Dataset(test_data, EvalConfig, tokenizer)
valid_set = T5Dataset(valid_data, EvalConfig, tokenizer)

def evaluate_peft_model(sample):
    # generate
    preds = model.generate(input_ids=torch.tensor(sample["input_ids"]).unsqueeze(dim=0).cuda(), 
                           do_sample=GenerateConfig.do_sample, 
                           top_k=GenerateConfig.top_k, 
                           top_p=GenerateConfig.top_p, 
                           temperature=GenerateConfig.temperature, 
                           repetition_penalty=GenerateConfig.repetition_penalty,
                           max_length=GenerateConfig.max_length)
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = sample['labels']
    # labels = torch.tensor([[0 if val == -100 else val for val in labels]]).cuda()
    # preds = torch.nn.functional.pad(preds, (0, labels.size(1) - preds.size(1)), 'constant', 0)
    model.eval()
    with torch.no_grad():
        output = model(input_ids = torch.tensor(sample["input_ids"]).unsqueeze(dim=0).cuda(), labels = torch.tensor([labels]).cuda())
        loss = output.loss.item()
    # print(loss)
    
    labels = torch.tensor([[0 if val == -100 else val for val in labels]]).cuda()
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return decoded_preds[0], decoded_labels[0], loss

predictions, references, losses = [], [], []
for sample in tqdm(test_set, desc="Generating"):
    p, l, loss = evaluate_peft_model(sample)
    predictions.append(p)
    references.append(l)
    # print("Predicted: ", p)
    # print("Reference: ", l)
    losses.append(loss)

result = {}
for metric_name in tqdm(EvalConfig.metrics, desc="Evaluating"):
    metric = metrics[metric_name]
    print(f"Computing {metric_name}...")
    
    if metric_name == "bleu":
        partial_result = metric.compute(predictions=predictions, references=[[r] for r in references])
        result["bleu-2"] = partial_result["precisions"][1] * 100
        result["bleu-4"] = partial_result["precisions"][3] * 100
    elif metric_name == "rouge":
        partial_result = metric.compute(predictions=predictions, references=references)
        result["rouge-L"] = partial_result["rougeL"] * 100
    elif metric_name == "perplexity":
        # partial_result = metric.compute(model_id='gpt2',
        #                  add_start_token=True,
        #                  predictions=predictions)
        partial_result = math.exp(sum(losses) / len(losses))
        result["perplexity"] = partial_result
    elif metric_name == "bertscore":
        partial_result = metric.compute(predictions=predictions, references=references, lang='en')
        result["bertscore_P"] = sum(partial_result["precision"]) / len(predictions)
        result["bertscore_R"] = sum(partial_result["recall"]) / len(predictions)
        result["bertscore_F1"] = sum(partial_result["f1"]) / len(predictions)

print(f"bleu-2: {result['bleu-2']}")
print(f"bleu-4: {result['bleu-4']}")
print(f"rouge-L: {result['rouge-L']}")
print(f"perplexity: {result['perplexity']}")
print(f"bertscore_P: {result['bertscore_P']}")
print(f"bertscore_R: {result['bertscore_R']}")
print(f"bertscore_F1: {result['bertscore_F1']}")