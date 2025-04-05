import os

from keras.src.saving import load_model
from transformers import AutoTokenizer

from model.preprocessor import preprocess_tweet


def load_bart_model():
    # Dynamically get the path of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path=base_dir+"\\model\\bert_model.keras";
    print("file_path::"+file_path)
    model = load_model(file_path)

    # Load tokenizer from the saved directory
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print("tokenizer::::",tokenizer("Test Tokenizer"))
    return model, tokenizer


def predict_bert_sentiment(question, model, tokenizer):
    """
       Predict the sentiment of a given tweet using an LSTM-based model.
       """
    input_ids, _ = preprocess_question(question)  # Only use `input_ids`

    # Predict Answer
    predictions = model.predict(input_ids)
    print(f"Raw BART Prediction: {predictions[0][0]}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Return the sentiment
    answer = (predictions > 0.5).astype(int)
    
    return sentiment[0][0]

