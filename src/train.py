import warnings
from utils.dataset import T5Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.utils import *
from datasets import load_dataset
import evaluate
import torch
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# TODO: add metrics -> bleu-2, rouge-L, perplexity.

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore")

ModelConfig = get_config('config/model.json')
TrainConfig = get_config('config/train.json')

tokenizer = AutoTokenizer.from_pretrained(ModelConfig.tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(ModelConfig.model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

raw_data = load_dataset("./dataset/esconv")
print("Successfully loaded dataset")
# metric = evaluate.load("sacrebleu")

train_data, test_data, valid_data = clean_data(raw_data)

train_set = T5Dataset(train_data, TrainConfig, tokenizer)
test_set = T5Dataset(test_data, TrainConfig, tokenizer)
valid_set = T5Dataset(valid_data[:10], TrainConfig, tokenizer)

print(f"Train dataset size: {len(train_set)}")
print(f"Test dataset size: {len(test_set)}")
print(f"Valid dataset size: {len(valid_set)}")

lora_config = LoraConfig(
    r=TrainConfig.lora['r'],
    lora_alpha=TrainConfig.lora['alpha'],
    target_modules=TrainConfig.lora['target_modules'],
    lora_dropout=TrainConfig.lora['dropout'],
    bias=TrainConfig.lora['bias'],
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)
training_args = Seq2SeqTrainingArguments(
    output_dir=TrainConfig.output_dir,
    seed=TrainConfig.seed,
    auto_find_batch_size=False,
    per_device_train_batch_size=TrainConfig.batch_size,
    per_device_eval_batch_size=TrainConfig.batch_size,
    evaluation_strategy = TrainConfig.evaluation_strategy,
    eval_steps=TrainConfig.eval_steps,
    learning_rate=TrainConfig.lr, # higher learning rate
    num_train_epochs=TrainConfig.n_epochs,
    logging_dir=f"{TrainConfig.output_dir}/logs",
    logging_strategy=TrainConfig.logging_strategy,
    logging_steps=TrainConfig.logging_steps,
    save_strategy=TrainConfig.save_strategy,
    save_steps=TrainConfig.save_steps,
    gradient_accumulation_steps=TrainConfig.gradient_accumulation_steps,
    weight_decay=TrainConfig.weight_decay,
    predict_with_generate=True,
    report_to=TrainConfig.report_to,
) 

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print(decoded_preds)
    # print(decoded_labels)
    
    result = {}
    for metric_name in TrainConfig.metrics:
        print(f"Computing {metric_name}...")
        metric = evaluate.load(path=f'./cached_metrics/{metric_name}')
        print(f"Loaded {metric_name}...")
        
        if metric_name == "bleu":
            partial_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result["bleu-2"] = partial_result["precisions"][1]
        elif metric_name == "rouge":
            partial_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result["rouge-L"] = partial_result["rougeL"]
        elif metric_name == "perplexity":
            partial_result = metric.compute(model_id='gpt2',
                             add_start_token=False,
                             predictions=decoded_preds)
            result["perplexity"] = partial_result["mean_perplexity"]

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
model.config.use_cache = False

trainer.train()

trainer.model.save_pretrained(TrainConfig.save_model_dir)
tokenizer.save_pretrained(TrainConfig.save_model_dir)