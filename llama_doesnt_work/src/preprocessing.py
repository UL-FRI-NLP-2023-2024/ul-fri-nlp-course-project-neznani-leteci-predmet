import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from functools import partial
from transformers import AutoTokenizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def create_prompt_formats(sample):
    """
    Creates a formatted prompt template for a prompt in the instruction dataset

    :param sample: Prompt or sample from the instruction dataset
    """
    CATEGORIES = ["Seminar", "Deliberation", "Social", "UX", "Procedure", "Imaginative entry", "Other" ]


    # Initialize static strings for the prompt template
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "Categorize the coversation as one of the 7 categories:\n\n" + "\n".join(CATEGORIES)
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    # Combine a prompt with the static strings
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n"
    input_context = f"{INPUT_KEY}\n{sample['Message']}" if sample["Message"] else None
    # if testing response is empty
    response = f"{RESPONSE_KEY}\n\n{sample['R2DiscussionType']}" 
    end = f"{END_KEY}"

    # Create a list of prompt template elements
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt = "\n\n".join(parts)

    # Store the formatted prompt template in a new key "text"
    sample["text"] = formatted_prompt

    return sample

def create_prompt_formats2(sample):
    """
    Creates a formatted prompt template for a prompt in the instruction dataset

    :param sample: Prompt or sample from the instruction dataset
    """
    CATEGORIES = ["Seminar", "Deliberation", "Social", "UX", "Procedure", "Imaginative entry", "Other" ]


    # Initialize static strings for the prompt template
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "Categorize the coversation as one of the 7 categories:\n\n" + "\n".join(CATEGORIES)
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    # Combine a prompt with the static strings
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n"
    input_context = f"{INPUT_KEY}\n{sample['Message']}" if sample["Message"] else None
    # if testing response is empty
    response = f"{RESPONSE_KEY}\n\n"

    # Create a list of prompt template elements
    parts = [part for part in [blurb, instruction, input_context, response] if part]

    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt = "\n\n".join(parts)

    # Store the formatted prompt template in a new key "text"
    sample["text"] = formatted_prompt

    return sample


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes dataset batch

    :param batch: Dataset batch
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        batch["text"],
        max_length = max_length,
        truncation = True,
    )

# Training dataset
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """
    Tokenizes dataset for fine-tuning

    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param seed: Random seed for reproducibility
    :param dataset (str): Instruction dataset
    """

    dataset = dataset.map(create_prompt_formats)

    print(dataset[0]['text'])

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched = True,
        remove_columns = ["Message", "R2DiscussionType", "text"],
    )

    # Filter out samples that have "input_ids" exceeding "max_length"
    # print size of dataset
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    #dataset = dataset.shuffle(seed = seed)

    return dataset

# Validation or test dataset
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """
    Tokenizes dataset for fine-tuning

    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param seed: Random seed for reproducibility
    :param dataset (str): Instruction dataset
    """

    dataset = dataset.map(create_prompt_formats2)

    print(dataset[0]['text'])

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched = True,
        remove_columns = ["Message", "R2DiscussionType", "text"],
    )

    # Filter out samples that have "input_ids" exceeding "max_length"
    # print size of dataset
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    #dataset = dataset.shuffle(seed = seed)

    return dataset