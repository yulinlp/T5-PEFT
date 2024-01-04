import time
import warnings
from utils.dataset import T5Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.utils import *
from datasets import load_dataset
import evaluate
import torch
import os
from transformers import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore")

ModelConfig = get_config('config/model.json')
TrainConfig = get_config('config/train.json')

if os.path.exists(TrainConfig.output_dir) and os.listdir(TrainConfig.output_dir):
    raise ValueError(f"Output directory ({TrainConfig.output_dir}) already exists and is not empty.")

if not os.path.exists(TrainConfig.logging_dir):
    os.mkdir(TrainConfig.logging_dir)

logger = create_logger(TrainConfig.logging_dir + '/' + time.strftime('%Y-%m-%d-%H-%M') + '.log')
logger.info(ModelConfig)
logger.info(TrainConfig)

metrics = {}
for metric_name in TrainConfig.metrics:
    metrics[metric_name] = evaluate.load(metric_name)
    logger.info(f"Loaded {metric_name}...")
    
tokenizer = AutoTokenizer.from_pretrained(ModelConfig.tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(ModelConfig.model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

raw_data = load_dataset("./dataset/esconv")
logger.info("Successfully loaded dataset")
# metric = evaluate.load("sacrebleu")

train_data, test_data, valid_data = clean_data(raw_data)

train_set = T5Dataset(train_data, TrainConfig, tokenizer)
test_set = T5Dataset(test_data, TrainConfig, tokenizer)
valid_set = T5Dataset(valid_data, TrainConfig, tokenizer)

logger.info(f"Train dataset size: {len(train_set)}")
logger.info(f"Test dataset size: {len(test_set)}")
logger.info(f"Valid dataset size: {len(valid_set)}")

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
    load_best_model_at_end=True
) 

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # print(preds, labels)
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    # print(decoded_preds)
    # print(decoded_labels)
    
    result = {}
    for metric_name in TrainConfig.metrics:
        logger.info(f"Computing {metric_name}...")
        metric = metrics[metric_name]
        # metric = evaluate.load(path=f'./cached_metrics/{metric_name}')        
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

    logger.info(f"bleu-2: {result['bleu-2']}")
    logger.info(f"rouge-L: {result['rouge-L']}")
    logger.info(f"perplexity: {result['perplexity']}")
    return result

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=TrainConfig.early_stopping_patience),]
)
model.config.use_cache = False

trainer.train()

trainer.model.save_pretrained(TrainConfig.save_model_dir)
tokenizer.save_pretrained(TrainConfig.save_model_dir)