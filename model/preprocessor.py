import re

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from model.LoadDataSet import load_data
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# Define a text cleaning function
def clean_text(text):
    """
    Clean text by removing URLs, mentions, hashtags, and special characters.
    """
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    return text.lower().strip()

# TBD need to add all the 

# Load your dataset

def training_dataset():
    print("TBD")


def input_data():
    print("TBD")

def get_preproceed_data():
    chem_data = load_data()

    print("TBD")

    return chem_data
def data_augmentation():
    tchem_data = load_data()

    chem_data = load_data()

    return {
        "missing_values": missing_values,
        "duplicate_count": duplicate_count,
        "descriptive_stats": descriptive_stats,
        "data_sample": twitter_data.head().to_dict(orient='records')  # Include a sample of the data
    }


def preprocess_chem_question(question, modeltokenizer, max_len=50):
    """
    Preprocess a question using the BERT tokenizer.
    """
    encoding = modeltokenizer.encode_plus(
        question,  # Input text
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors="tf"  # Return TensorFlow tensors
    )
    return encoding['input_ids'], encoding['attention_mask']
