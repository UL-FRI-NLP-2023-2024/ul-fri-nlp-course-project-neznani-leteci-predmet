import numpy as np
import torch

from torch.utils.data import DataLoader


from src.data_loader import load_dataset, extract_data
from src.preprocessing import preprocess_text
from src.model import load_pretrained_bert_model, tokenize_text, fine_tune_bert, evaluate_model, split_train_val_data, print_evaluation_metrics

# Load dataset
file_path = 'data/CollabWriteAnalysisCountCodesLadyorTigerF20nS21S22wGSAnalysis27Jan2023_14Mar2024CleanF.xlsm'
df = load_dataset(file_path)

df = df.groupby('R2DiscussionType').filter(lambda x: len(x) >= 5)
df = df[['Message', 'R2DiscussionType']]
# Preprocess text
df['preprocessed_text'] = df['Message'].apply(preprocess_text)
preprocessed_text_list = df['preprocessed_text'].tolist()

# Load pre-trained BERT model
model_name = 'bert-base-uncased'
num_labels = len(df['R2DiscussionType'].unique())
model, tokenizer = load_pretrained_bert_model(model_name, num_labels)

# Tokenize text
max_length = 32  # Example, adjust based on your input size
inputs = tokenize_text(tokenizer, preprocessed_text_list, max_length)

# Add message back to inputs
messeges = df['Message'].tolist()
inputs['messages'] = messeges
# Split into train and val datasets - Verjetno ok - ne vem, kako bi to tocno testiral
train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks, train_messages, val_messages = split_train_val_data(inputs, df)
# print training data and validation data
#print(train_inputs)
#print("-"*100)
#print(val_inputs)

# Split val_inputs, val_labels, val_masks and val_messages into val and test (50 50)
val_inputs, test_inputs = np.array_split(val_inputs, 2)
val_labels, test_labels = np.array_split(val_labels, 2)
val_masks, test_masks = np.array_split(val_masks, 2)
val_messages, test_messages = np.array_split(val_messages, 2)



# Transform labels to tensors
unique_labels = df['R2DiscussionType'].unique()
label_map = {label: torch.tensor(i) for i, label in enumerate(unique_labels)}
print(label_map)
train_labels = [label_map[label] for label in train_labels]
val_labels = [label_map[label] for label in val_labels]
test_labels = [label_map[label] for label in test_labels]


train_labels = torch.stack(train_labels)
val_labels = torch.stack(val_labels)
test_labels = torch.stack(test_labels)

# Create DataLoader
train_dataset = [(input_ids, attention_mask, label) for input_ids, attention_mask, label in zip(train_inputs, train_masks, train_labels)]
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = [(input_ids, attention_mask, label) for input_ids, attention_mask, label in zip(val_inputs, val_masks, val_labels)]
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Fine tune
num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)

#model = fine_tune_bert(model, train_dataloader, val_dataloader, num_epochs, num_training_steps)

# Save model - Na github sem dodal samo mapo models kamor lahko pol lokalno shranjujemo (modela nisem nalagal ker je huge)
#model_path = 'models/bert_model.pt'
#torch.save(model, model_path)
# Load dataset
model = torch.load('models/bert_model.pt')

# Evaluate model on the test dataset
test_dataset = [(input_ids, attention_mask, label) for input_ids, attention_mask, label in zip(test_inputs, test_masks, test_labels)]
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


metrics = evaluate_model(model, test_dataloader, "cuda", test_messages, label_map)
print_evaluation_metrics(metrics)

