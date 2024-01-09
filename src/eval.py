import argparse
import math
import time
import torch
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import warnings
import os
from datasets import load_dataset
from random import randrange
from utils.dataset import *
from utils.utils import *
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/t5-base/', help='config directory')
# ConfigRoot = 'config/gpt2-small/'
# ConfigRoot = 'config/t5-base/'
# ConfigRoot = 'config/flan-t5-base/'

args = parser.parse_args()
ConfigRoot = args.config

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore")

ModelConfig = get_config(ConfigRoot + 'model.json')
EvalConfig = get_config(ConfigRoot + 'eval.json')
GenerateConfig = EasyDict(EvalConfig.GenerateConfig)
seed_everything(EvalConfig.seed)

if not os.path.exists(EvalConfig.logging_dir):
    os.mkdir(EvalConfig.logging_dir)

logger = create_logger(EvalConfig.logging_dir + '/' + time.strftime('%Y-%m-%d-%H-%M') + '.log')
logger.info(ModelConfig)
logger.info(EvalConfig)

metrics = {}
for metric_name in EvalConfig.metrics:
    metrics[metric_name] = evaluate.load('cached_metrics/' + metric_name)
    logger.info(f"Loaded {metric_name}...")

if 'gpt' in ModelConfig.model_name.lower():
    model = AutoModelForCausalLM.from_pretrained(ModelConfig.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.tokenizer_name)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(ModelConfig.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.tokenizer_name)

if EvalConfig.is_PEFT:
    # Load peft config for pre-trained checkpoint etc.
    peft_model_id = EvalConfig.PEFT_model_dir
    model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
    if 'gpt' in ModelConfig.model_name.lower():
        # load base LLM model and tokenizer
        bos = '<|bos|>'
        pad = '<|pad|>'
        user = '<|user|>'
        assistant = '<|assistant|>'
        special_tokens_dict = {'bos_token': bos, 'pad_token': pad, 'additional_special_tokens': [user, assistant]}
    else:
        bos = '<|bos|>'
        user = '<|user|>'
        assistant = '<|assistant|>'
        special_tokens_dict = {'bos_token': bos, 'additional_special_tokens': [user, assistant]}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()
    logger.info("PEFT Model loaded") 
else:
    logger.info("BASE Model loaded") 
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.config.use_cache = True
model.eval()

raw_data = load_dataset("./dataset/esconv")
logger.info("Successfully loaded dataset")

train_data, test_data, valid_data = clean_data(raw_data, EvalConfig.data_official)
if 'gpt' in ModelConfig.model_name.lower():
    train_set = GPT2Dataset(train_data, EvalConfig, ModelConfig, tokenizer)
    test_set = GPT2Dataset(test_data, EvalConfig, ModelConfig, tokenizer)
    valid_set = GPT2Dataset(valid_data, EvalConfig, ModelConfig, tokenizer)
else:
    train_set = T5Dataset(train_data, EvalConfig, ModelConfig, tokenizer)
    test_set = T5Dataset(test_data, EvalConfig, ModelConfig, tokenizer)
    valid_set = T5Dataset(valid_data, EvalConfig, ModelConfig, tokenizer)

def evaluate_gpt2(sample):
    # generate
    target_idx = find_last_index(sample['input_ids'], tokenizer.additional_special_tokens_ids[1])
    raw_input_ids = sample['input_ids']
    label_ids = sample['input_ids'][target_idx + 1:]
    sample['input_ids'] = [tokenizer.pad_token_id] * (len(sample['input_ids']) - target_idx - 1) + sample['input_ids'][:target_idx + 1]
    preds = model.generate(input_ids=torch.tensor(sample["input_ids"]).unsqueeze(dim=0).cuda(), 
                           do_sample=GenerateConfig.do_sample, 
                           pad_token_id=tokenizer.pad_token_id,
                           top_k=GenerateConfig.top_k, 
                           top_p=GenerateConfig.top_p, 
                           temperature=GenerateConfig.temperature, 
                           repetition_penalty=GenerateConfig.repetition_penalty,
                           max_new_tokens=GenerateConfig.max_length)
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_inputs = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    # print("decoded_inputs: ", decoded_inputs)
    # exit()
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = decoded_preds[0].replace(decoded_inputs, '')
    
    # Replace -100 in the labels as we can't decode them.
    with torch.no_grad():
        output = model(input_ids = torch.tensor(raw_input_ids).unsqueeze(dim=0).cuda(), 
                       labels = torch.tensor([-100 if i == tokenizer.pad_token_id else i for i in raw_input_ids]).unsqueeze(dim=0).cuda())
        loss = output.loss.item()

    decoded_labels = tokenizer.batch_decode([label_ids], skip_special_tokens=True)[0]
    return decoded_preds, decoded_labels, loss

def evaluate_t5(sample):
    # target_idx = sample['input_ids'].index(tokenizer.eos_token_id)
    # sample['input_ids'] = sample['input_ids'][ :target_idx] + [tokenizer.additional_special_tokens_ids[1]] + [tokenizer.pad_token_id] * (len(sample['input_ids']) - target_idx - 1)
    preds = model.generate(input_ids=torch.tensor(sample["input_ids"]).unsqueeze(dim=0).cuda(), 
                           do_sample=GenerateConfig.do_sample, 
                           top_k=GenerateConfig.top_k, 
                           top_p=GenerateConfig.top_p, 
                           temperature=GenerateConfig.temperature, 
                           repetition_penalty=GenerateConfig.repetition_penalty,
                           max_length=GenerateConfig.max_length)
    
    if isinstance(preds, tuple):
        preds = preds[0]
    # decoded_inputs = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    # logger.info(decoded_inputs)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # logger.info(decoded_preds)

    labels = sample['labels']
    with torch.no_grad():
        output = model(input_ids = torch.tensor(sample["input_ids"]).unsqueeze(dim=0).cuda(), 
                       labels = torch.tensor([labels]).cuda())
        loss = output.loss.item()
    
    # Replace -100 in the labels as we can't decode them.
    labels = torch.tensor([[0 if val == -100 else val for val in labels]]).cuda()
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return decoded_preds[0], decoded_labels[0], loss

logger.info("="*30)
logger.info("="*30)
logger.info("Evaluation started")
logger.info("="*30)
logger.info("="*30)

predictions, references, losses = [], [], []
for idx, sample in tqdm(enumerate(test_set), desc="Generating"):
    if 't5' in ModelConfig.model_name.lower():
        p, l, loss = evaluate_t5(sample)
    else:
        p, l, loss = evaluate_gpt2(sample)
    predictions.append(p)
    references.append(l)
    losses.append(loss)
    
    logger.info(f"{idx} Predicted: {p}")
    logger.info(f"{idx} Reference: {l}")
    logger.info(f"{idx} Loss: {loss}")
    logger.info("#" * 30)
    # exit()
    
result = {}
for metric_name in tqdm(EvalConfig.metrics, desc="Evaluating"):
    metric = metrics[metric_name]
    logger.info(f"Computing {metric_name}...")
    
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

logger.info(f"bleu-2: {result['bleu-2']}")
logger.info(f"bleu-4: {result['bleu-4']}")
logger.info(f"rouge-L: {result['rouge-L']}")
logger.info(f"perplexity: {result['perplexity']}")
logger.info(f"bertscore_P: {result['bertscore_P']}")
logger.info(f"bertscore_R: {result['bertscore_R']}")
logger.info(f"bertscore_F1: {result['bertscore_F1']}")