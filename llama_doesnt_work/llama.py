import os
from random import randrange
from functools import partial
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          pipeline,
                          logging,
                          set_seed)

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
from trl import SFTTrainer

from src.preprocessing import preprocess_dataset, create_prompt_formats, preprocess_dataset2
from src.model import create_bnb_config, load_model, get_max_length, fine_tune
from src.data_loader import load_dataset_file
from src.evaluation import evaluate_model, get_predictions

from datasets import Dataset


# Add your API key here from the Hugging Face website (need access to meta-llama model)
auth_token = ""

# Load dataset
file_path = 'data/CollabWriteAnalysisCountCodesLadyorTigerF20nS21S22wGSAnalysis27Jan2023_14Mar2024CleanF.xlsm'
df = load_dataset_file(file_path)

# Count the number of unique labels
num_labels = len(df['R2DiscussionType'].unique())

# Delete rows with less than 5 occurrences of R2DiscussionType
df = df.groupby('R2DiscussionType').filter(lambda x: len(x) >= 5)

# Select only the 'Message' and 'R2DiscussionType' columns
df = df[['Message', 'R2DiscussionType']]

# Split into training and validation sets
train_size = 0.8
train_df = df.sample(frac=train_size, random_state=42)
val_df = df.drop(train_df.index)

# Reset the index
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Create a Hugging Face dataset
dataset = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)


# Print the number of prompts, column names, and a random prompt
#print(f'Number of prompts: {len(dataset)}')
#print(f'Column names are: {dataset.column_names}')
#print(dataset[randrange(len(dataset))])


# Load model
# this model didn't work
#model_name = 'daryl149/llama-2-7b-chat-hf'
model_name = 'meta-llama/Llama-2-7b-hf'
load_in_4bit = True
bnb_4bit_use_double_quant = True
bnb_4bit_quant_type = 'nf4'
bnb_4bit_compute_dtype = 'float16'

bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)
model, tokenizer = load_model(model_name, bnb_config, auth_token)



# Get the maximum token length
max_length = get_max_length(model)

# Preprocess dataset
seed = 42
# Training dataset has different format than validation dataset
# training dataset has 'R2DiscussionType' column while validation dataset does not, since the model need to predict the 'R2DiscussionType' column
preprocessed_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)
preprocessed_dataset_val = preprocess_dataset2(tokenizer, max_length, seed, dataset_val)



# Print the preprocessed dataset and the first sample
print(preprocessed_dataset)
print(preprocessed_dataset[0])


# Fine-tune the model


## QLoRA configuration
# LoRA attention dimension
lora_r = 16

# Alpha parameter for LoRA scaling
lora_alpha = 64

# Dropout probability for LoRA layers
lora_dropout = 0.1

# Bias
bias = "none"

# Task type
task_type = "CAUSAL_LM"


## Training arguments parameters
# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Batch size per GPU for training
per_device_train_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Optimizer to use
optim = "paged_adamw_32bit"

# Number of training steps (overrides num_train_epochs)
max_steps = 20

# Linear warmup steps from 0 to learning_rate
warmup_steps = 2

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True

# Log every X updates steps
logging_steps = 1

# Fine tune the model
model, tokenizer, trainer = fine_tune(model,
          tokenizer,
          preprocessed_dataset,
          lora_r,
          lora_alpha,
          lora_dropout,
          bias,
          task_type,
          per_device_train_batch_size,
          gradient_accumulation_steps,
          warmup_steps,
          max_steps,
          learning_rate,
          fp16,
          logging_steps,
          output_dir,
          optim)



# Evaluate the model
model.eval()

metrics = evaluate_model(model, preprocessed_dataset_val, "cpu", tokenizer)
print(metrics)





# Get predictions
# This is where our implementation stops working
# While running this code on hpc, program just run forever and never stops
# Sometimes this happens after fine tuning the model other time after evaluating the model

device = "cuda" if torch.cuda.is_available() else "cpu"
predictions = get_predictions(model, preprocessed_dataset_val, device, tokenizer)

for i, prediction in enumerate(predictions):
    print(f"Prediction {i}: {prediction}")
print("Done!")