import time
import warnings
from utils.dataset import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from utils.utils import *
from datasets import load_dataset
import evaluate
import torch
import os
from transformers import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType

# ConfigRoot = 'config/gpt2-small/'
# ConfigRoot = 'config/t5-base/'
ConfigRoot = 'config/flan-t5-base/'

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore")

ModelConfig = get_config(ConfigRoot + 'model.json')
TrainConfig = get_config(ConfigRoot + 'train.json')

seed_everything(TrainConfig.seed)

# if os.path.exists(TrainConfig.output_dir) and os.listdir(TrainConfig.output_dir):
#     raise ValueError(f"Output directory ({TrainConfig.output_dir}) already exists and is not empty.")

if not os.path.exists(TrainConfig.logging_dir):
    os.mkdir(TrainConfig.logging_dir)

logger = create_logger(TrainConfig.logging_dir + '/' + time.strftime('%Y-%m-%d-%H-%M') + '.log')
logger.info(ModelConfig)
logger.info(TrainConfig)
    
if 'gpt' in ModelConfig.model_name:
    model = AutoModelForCausalLM.from_pretrained(ModelConfig.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.tokenizer_name, padding_side="right")
    bos = '<|bos|>'
    pad = '<|pad|>'
    user = '<|user|>'
    assistant = '<|assistant|>'
    special_tokens_dict = {'bos_token': bos, 'pad_token': pad, 'additional_special_tokens': [user, assistant]}
    tokenizer.add_special_tokens(special_tokens_dict)
    # tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(ModelConfig.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.tokenizer_name)
    user = '<|user|>'
    assistant = '<|assistant|>'
    special_tokens_dict = {'additional_special_tokens': [user, assistant]}
    tokenizer.add_special_tokens(special_tokens_dict)
    # tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Backbone loaded") 

raw_data = load_dataset("./dataset/esconv")
logger.info("Successfully loaded dataset")

train_data, test_data, valid_data = clean_data(raw_data, TrainConfig.data_official)
if 'gpt' in ModelConfig.model_name:
    train_set = GPT2Dataset(train_data, TrainConfig, ModelConfig, tokenizer)
    test_set = GPT2Dataset(test_data, TrainConfig, ModelConfig, tokenizer)
    valid_set = GPT2Dataset(valid_data, TrainConfig, ModelConfig, tokenizer)
else:
    train_set = T5Dataset(train_data, TrainConfig, ModelConfig, tokenizer)
    test_set = T5Dataset(test_data, TrainConfig, ModelConfig, tokenizer)
    valid_set = T5Dataset(valid_data, TrainConfig, ModelConfig, tokenizer)

# exit()
logger.info(f"Train dataset size: {len(train_set)}")
logger.info(f"Test dataset size: {len(test_set)}")
logger.info(f"Valid dataset size: {len(valid_set)}")

lora_config = LoraConfig(
    r=TrainConfig.lora['r'],
    lora_alpha=TrainConfig.lora['alpha'],
    lora_dropout=TrainConfig.lora['dropout'],
    bias=TrainConfig.lora['bias'],
    task_type=TaskType.SEQ_2_SEQ_LM if 't5' in ModelConfig.model_name else TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collator
if 'gpt' in ModelConfig.model_name:
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
else:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8
    )

training_args = TrainingArguments(
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
    report_to=TrainConfig.report_to,
    load_best_model_at_end=True
) 

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_set,
    eval_dataset=test_set,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=TrainConfig.early_stopping_patience),]
)
model.config.use_cache = False

trainer.train()

trainer.model.save_pretrained(TrainConfig.save_model_dir)
tokenizer.save_pretrained(TrainConfig.save_model_dir)