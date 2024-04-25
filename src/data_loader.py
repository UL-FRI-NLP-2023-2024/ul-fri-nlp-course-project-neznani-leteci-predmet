import pandas as pd

def load_dataset(file_path, sheet_name='Combined Discussions'):
    """
    Load the dataset from the Excel file.
    
    Args:
    file_path (str): Path to the Excel file.
    sheet_name (str): Name of the sheet containing the data.
    
    Returns:
    pd.DataFrame: DataFrame containing the dataset.
    """
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def extract_data(df, columns=['R2DiscussionType', 'R2Uptake']):
    """
    Extract relevant data from the DataFrame.
    
    Args:
    df (pd.DataFrame): DataFrame containing the dataset.
    columns (list): List of column names to extract.
    
    Returns:
    pd.DataFrame: DataFrame containing the extracted data.
    """
    # Extract relevant columns
    extracted_data = df[columns]
    return extracted_data