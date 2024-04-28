import numpy as np
import torch

from torch.utils.data import DataLoader


from src.data_loader import load_dataset, extract_data
from src.preprocessing import preprocess_text
from src.model import load_pretrained_bert_model, tokenize_text, fine_tune_bert, evaluate_model, split_train_val_data, print_evaluation_metrics

# Load dataset
file_path = 'data/CollabWriteAnalysisCountCodesLadyorTigerF20nS21S22wGSAnalysis27Jan2023_14Mar2024CleanF.xlsm'
df = load_dataset(file_path)

# Preprocess text
df['preprocessed_text'] = df['Message'].apply(preprocess_text)
preprocessed_text_list = df['preprocessed_text'].tolist()

# Load pre-trained BERT model
model_name = 'bert-base-uncased'
num_labels = len(df['R2DiscussionType'].unique())
model, tokenizer = load_pretrained_bert_model(model_name, num_labels)

# Tokenize text
max_length = 128  # Example, adjust based on your input size
inputs = tokenize_text(tokenizer, preprocessed_text_list, max_length)

# Split into train and val datasets - Verjetno ok - ne vem, kako bi to tocno testiral
train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = split_train_val_data(inputs, df)
# print training data and validation data
print(train_inputs)
print("-"*100)
print(val_inputs)

# Transform labels to tensors
unique_labels = df['R2DiscussionType'].unique()
label_map = {label: torch.tensor(i) for i, label in enumerate(unique_labels)}
print(label_map)
train_labels = [label_map[label] for label in train_labels]
val_labels = [label_map[label] for label in val_labels]


train_labels = torch.stack(train_labels)
val_labels = torch.stack(val_labels)

# Create DataLoader
train_dataset = [(input_ids, attention_mask, label) for input_ids, attention_mask, label in zip(train_inputs, train_masks, train_labels)]
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = [(input_ids, attention_mask, label) for input_ids, attention_mask, label in zip(val_inputs, val_masks, val_labels)]
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Fine tune
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

model = fine_tune_bert(model, train_dataloader, val_dataloader, num_epochs, num_training_steps)

# Save model - Na github sem dodal samo mapo models kamor lahko pol lokalno shranjujemo (modela nisem nalagal ker je huge)
model_path = 'models/bert_model.pt'
torch.save(model, model_path)
# Load dataset
#model = torch.load('models/bert_model.pt')

# Evaluate model
metrics = evaluate_model(model, val_dataloader, "cpu")
print_evaluation_metrics(metrics)