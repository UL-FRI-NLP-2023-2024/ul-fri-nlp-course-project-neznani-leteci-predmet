import numpy as np
import torch

import argparse
from torch.utils.data import DataLoader


from src.data_loader import load_dataset, extract_data
from src.preprocessing import preprocess_text
from src.model import load_pretrained_bert_model, tokenize_text, fine_tune_bert, evaluate_model, split_train_val_data, print_evaluation_metrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load dataset
def run_bert(file_path):
    #file_path = 'data/CollabWriteAnalysisCountCodesLadyorTigerF20nS21S22wGSAnalysis27Jan2023_14Mar2024CleanF.xlsm'
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
    
# Traditional method - TFIDF
def run_tfidf(file_path):
    # Load dataset
    df = load_dataset(file_path)
    
    # Filter out classes with less than 5 instances
    df = df.groupby('R2DiscussionType').filter(lambda x: len(x) >= 5)
    df = df[['Message', 'R2DiscussionType']]
    
    # Preprocess text
    df['preprocessed_text'] = df['Message'].apply(preprocess_text)
    preprocessed_text_list = df['preprocessed_text'].tolist()
    
    # Convert text data to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=5000)
    x = vectorizer.fit_transform(preprocessed_text_list)
    y =df['R2DiscussionType'].values
    
    # Split the dataset into train, validation and test sets
    # x_train and y_train - training data and labels - 60%
    # x_temp and y_temp - temporary data and labels - 40% - later split into validation and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
    # Validation - x_val and y_val and Test - x_test and y_test
    X_val, X_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Hyperparameter tuning using GridSearchCV with stratified k-fold cross-validation
    param_grid = {
        # 'C' is the inverse of regularization strength - positive float, higher value means fit the training data more closely	
        'C': [0.1, 1, 10, 100],
        # l2 - Ridge, l1 - Lasso
        'penalty': ['l2'],
        'solver': ['liblinear']
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=skf)
    grid_search.fit(x_train, y_train)
    print(f'Best parameters: {grid_search.best_params_}')
    
    # Train the logistic Regression classifier with the best hyperparameters
    best_params = grid_search.best_params_
    model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver=best_params['solver'])
    model.fit(x_train, y_train)
    
    # Evaluate the model on the validation set
    y_pred = model.predict(X_val)
    report_val = classification_report(y_val, y_pred, zero_division=1, digits=4)
    print("Validation set evaluation:")
    print(report_val)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    report_test = classification_report(y_test, y_pred, zero_division=1, digits=4)
    print("Test set evaluation:")
    print(report_test)
    
    # Save classification report to a file
    with open('results/tfidf/tfidf_classification_report_test.out', 'w') as f:
        # Write parameters and classification report to the file
        f.write(f'Parameters used: {best_params}\n\n')
        f.write("Validation set evaluation:\n")
        f.write(report_val + "\n\n")
        f.write("Test set evaluation:\n")
        f.write(report_test + "\n\n")
        
    # Save the model
    model_path = 'models/tfidf_classifier.pkl'
    joblib.dump(model, model_path)
    
    # Save the vectorizer
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_path)
    
    
if __name__ == '__main__':
    file_path = 'data/CollabWriteAnalysisCountCodesLadyorTigerF20nS21S22wGSAnalysis27Jan2023_14Mar2024CleanF.xlsm'
    parser = argparse.ArgumentParser(description='Run text classification using BERT or TF-IDF')
    parser.add_argument('--model', type=str, choices=['bert', 'tfidf'], required=True, help='Model to run: "bert" or "tfidf"')
    
    args = parser.parse_args()
    
    if args.model == 'bert':
        run_bert(file_path)
    elif args.model == 'tfidf':
        run_tfidf(file_path)
    else:
        print('Invalid model choice. Choose "bert" or "tfidf"')

