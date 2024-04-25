import numpy as np
import torch

from src.data_loader import load_dataset, extract_data
from src.preprocessing import preprocess_text
from src.model import load_pretrained_bert_model, tokenize_text, fine_tune_bert, evaluate_model

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

# Split into train and val datasets TODO
# Fine tune TODO
# Evaluate TODO