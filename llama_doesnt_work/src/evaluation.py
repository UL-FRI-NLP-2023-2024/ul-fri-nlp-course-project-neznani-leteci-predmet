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
                          DataCollatorWithPadding,
                          EarlyStoppingCallback,
                          pipeline,
                          logging,
                          set_seed)

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
from trl import SFTTrainer
from torch.utils.data import DataLoader


def evaluate_model(model, dataset, device, tokenizer):
    """
    Evaluates the fine-tuned model on the validation dataset.

    :param model: The fine-tuned model.
    :param dataset: The validation dataset.
    :param device: The device to run the evaluation on ('cpu' or 'cuda').
    :return: A dictionary with evaluation metrics (e.g., perplexity).
    """
    # Define the evaluation arguments
    eval_args = TrainingArguments(
        per_device_eval_batch_size=1,
        output_dir='./results',
        do_train=False,
        do_eval=True,
        no_cuda=device == 'cpu',
        logging_steps=1,
        seed=42,
        disable_tqdm=False,
        fp16=True if device == 'cuda' else False,
    )

    # Create the Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Perform the evaluation
    eval_result = trainer.evaluate()

    # Compute perplexity
    perplexity = torch.exp(torch.tensor(eval_result['eval_loss']))
    eval_result['perplexity'] = perplexity.item()

    print(eval_result)

    return eval_result

def get_predictions(model, dataset, device, tokenizer):
    """
    Get model predictions on a validation dataset

    :param model (torch.nn.Module): The fine-tuned model
    :param dataset (torch.utils.data.Dataset): The validation dataset containing input_ids and attention_mask
    :param device (str): The device to run the evaluation on ('cpu' or 'cuda')
    :return: List of predictions
    """
    # Ensure the model is on the correct device
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=1)
    predictions = []
    i = 0
    with torch.no_grad():
        for batch in dataloader:
            print(i)
            i += 1
            # Access input_ids and attention_mask lists from the batch
            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)

            # Get model predictions
            outputs = model.generate(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))

            # Decode the predictions
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_preds)

    return predictions