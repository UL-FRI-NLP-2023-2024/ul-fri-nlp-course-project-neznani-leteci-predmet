import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from .preprocessing import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

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
def split_train_val_data(inputs, df, val_split=0.4):
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

    # Get messages from the tokenized inputs
    messages = inputs['messages']
    
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
    
    
    # Split messages into training and validation sets
    train_messages, val_messages, _, _ = train_test_split(
        messages,
        input_ids,
        test_size=val_split,
        random_state=42)
    
    # Return split data
    return train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks, train_messages, val_messages
    

def fine_tune_bert(model, train_dataloader, val_dataloader, num_epochs, num_training_steps, lr=5e-5):
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
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_dataloader):
            print(f'Batch {i+1}/{len(train_dataloader)}')
            # Unpack the batch
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()


        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                print(f'Validation Batch {i+1}/{len(val_dataloader)}')
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
        
    return model

def evaluate_model(model, dataloader, device, val_messages, label_map):
    """
    Evaluate the fine-tuned BERT model on the validation dataset.
    
    Args:
    model (BertForSequenceClassification): Fine-tuned BERT model.
    dataloader (DataLoader): DataLoader for the validation dataset.
    device (str): Device to use for evaluation ('cpu' or 'cuda').
    
    Returns:
    dict: A dictionary containing average loss and additional evaluation metrics on the validation dataset.
    """

    model.eval()
    total_loss = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            # Append predictions and true labels
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # open file analysis.txt

    file = open("analysis.txt", "w")
    for i in range(len(all_predictions)):
        # use label_map to get the label from the label index and true label
        # this is how label_map looks like: {'Social': tensor(0), 'Seminar': tensor(1), 'Procedure': tensor(2), 'Other': tensor(3), 'Deliberation': tensor(4), 'UX': tensor(5), 'Imaginative entry': tensor(6)}
        all_predictions[i] = list(label_map.keys())[list(label_map.values()).index(all_predictions[i])]
        all_true_labels[i] = list(label_map.keys())[list(label_map.values()).index(all_true_labels[i])]
        #print(f'Prediction: {all_predictions[i]}, True Label: {all_true_labels[i]}, Message: {val_messages[i]}')
        # save this to file analysis.txt
        file.write(f'Prediction: {all_predictions[i]}, True Label: {all_true_labels[i]}, Message: {val_messages[i]}\n')
    file.close()

            
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    # Calculate additional evaluation metrics
    # Accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    
    # Precision, Recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted')
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    
    # Classification report
    class_report = classification_report(all_true_labels, all_predictions)
    
    # Return evaluation metrics
    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    
def print_evaluation_metrics(metrics):
    """
    Print the evaluation metrics for the fine-tuned BERT model.
    
    Args:
    metrics (dict): A dictionary containing evaluation metrics.
    """
    print(f'Average Loss: {metrics["avg_loss"]}')
    print(f'Accuracy: {metrics["accuracy"]}')
    print(f'Precision: {metrics["precision"]}')
    print(f'Recall: {metrics["recall"]}')
    print(f'F1 Score: {metrics["f1"]}')
    print('Confusion Matrix:')
    print(metrics["confusion_matrix"])
    print()
    print('Classification Report:')
    print(metrics["classification_report"])