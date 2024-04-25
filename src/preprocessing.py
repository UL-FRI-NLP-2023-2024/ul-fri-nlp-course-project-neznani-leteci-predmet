import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text, lowercase=True, remove_punctuation=True, remove_stopwords=True, stemming=True, lemmatization=True):
    """
    Preprocess the text data.
    
    Args:
    text (str): Input text.
    lowercase (bool): Whether to convert text to lowercase.
    remove_punctuation (bool): Whether to remove punctuation.
    remove_stopwords (bool): Whether to remove stopwords.
    stemming (bool): Whether to perform stemming.
    lemmatization (bool): Whether to perform lemmatization.
    
    Returns:
    str: Preprocessed text.
    """
    # Tokenization
    tokens = word_tokenize(text)
    
    # Apply preprocessing steps based on options
    if lowercase:
        tokens = [word.lower() for word in tokens]
    if remove_punctuation:
        tokens = [word for word in tokens if word not in string.punctuation]
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text
