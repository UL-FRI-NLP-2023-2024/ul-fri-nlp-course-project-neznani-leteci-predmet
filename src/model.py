import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from .preprocessing import preprocess_text
from sklearn.model_selection import train_test_split

def load_pretrained_bert_model(model_name, num_labels):
    """
    Load a pre-trained BERT model for sequence classification.
    
    Args:
    model_name (str): Name of the pre-trained BERT model.
    num_labels (int): Number of output labels for the classification task.
    
    Returns:
    BertForSequenceClassification: Pre-trained BERT model for sequence classification.
    BertTokenizer: BERT tokenizer.
    """
    # Load pre-trained BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def tokenize_text(tokenizer, text, max_length):
    """
    Tokenize the input text using the BERT tokenizer.
    
    Args:
    tokenizer (BertTokenizer): BERT tokenizer.
    text (str or List[str]): Input text or list of input texts.
    max_length (int): Maximum sequence length.
    
    Returns:
    dict: Dictionary containing tokenized inputs.
    """
    # Tokenize input text
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'  # Return PyTorch tensors
    )
    return inputs


# Split the tokenized inputs and labels into training and validation datasets.
# Zaenkrat 10% val in 90% train
def split_train_val_data(inputs, df, val_split=0.1):
    """
    Split the tokenized inputs and labels into training and validation datasets.
    
    Args:
    inputs (dict): Dictionary containing tokenized inputs.
    labels (List[int]): List of labels.
    val_split (float): Fraction of validation data.
    """
    
    # Get input_ids from tokenized inputs
    input_ids = inputs['input_ids']
    
    # Get attention_mask from tokenized inputs
    attention_mask = inputs['attention_mask']
    
    # Get labels from the DataFrame
    labels = df['R2DiscussionType'].values
    
    # Split input_ids and labels into training and validation sets
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_ids,
        labels, 
        random_state=42,
        test_size=val_split)
    
    # Split attention_mask into training and validation sets
    train_masks, val_masks, _, _ = train_test_split(
        attention_mask, 
        input_ids, 
        test_size=val_split, 
        random_state=42)
    
    # Return split data
    return train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks
    

def fine_tune_bert(model, train_dataloader, val_dataloader, num_epochs, num_training_steps):
    """
    Fine-tune the pre-trained BERT model on the training dataset.
    
    Args:
    model (BertForSequenceClassification): Pre-trained BERT model.
    train_dataloader (DataLoader): DataLoader for the training dataset.
    val_dataloader (DataLoader): DataLoader for the validation dataset.
    num_epochs (int): Number of training epochs.
    num_training_steps (int): Number of training steps in each epoch.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            # Unpack the batch
            input_ids = batch[0][0].to(device)
            attention_mask = batch[0][1].to(device)
            token_type_ids = batch[0][2].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()

            print(input_ids.shape)
            print(attention_mask.shape)
            print(token_type_ids.shape)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()


        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch[0]  
                labels = batch[1] 

                input_ids = inputs[0].to(device)
                attention_mask = inputs[1].to(device)
                token_type_ids = inputs[2].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')

def evaluate_model(model, dataloader, device):
    """
    Evaluate the fine-tuned BERT model on the validation dataset.
    
    Args:
    model (BertForSequenceClassification): Fine-tuned BERT model.
    dataloader (DataLoader): DataLoader for the validation dataset.
    device (str): Device to use for evaluation ('cpu' or 'cuda').
    
    Returns:
    float: Average loss on the validation dataset.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss