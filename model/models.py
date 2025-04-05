import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from transformers import AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification

twitter_data = pd.read_csv('/app/data/twitter/training.1600000.processed.noemoticon.csv', encoding='latin-1', names=['sentiment', 'id', 'date', 'flag', 'user', 'text'])

# Define text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#\w+", "", text)     # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    return text.lower().strip()

# Load dataset and preprocess
twitter_data = twitter_data[['sentiment', 'text']]
twitter_data['sentiment'] = twitter_data['sentiment'].map({0: 0, 4: 1})
twitter_data = twitter_data.dropna(subset=['text'])
twitter_data['text'] = twitter_data['text'].astype(str).apply(clean_text)

# Split dataset
x_text = twitter_data['text'].tolist()
y_labels = twitter_data['sentiment'].tolist()
x_train, x_val, y_train, y_val = train_test_split(x_text, y_labels, test_size=0.2, random_state=42)

# Define LSTM model
def create_lstm_model(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Tokenize and pad for LSTM
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_val_seq = tokenizer.texts_to_sequences(x_val)
max_len = 50
x_train_padded = pad_sequences(x_train_seq, maxlen=max_len, padding='post', truncating='post')
x_val_padded = pad_sequences(x_val_seq, maxlen=max_len, padding='post', truncating='post')

x_train_padded = tf.convert_to_tensor(x_train_padded, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_val_padded = tf.convert_to_tensor(x_val_padded, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

# Train LSTM model
lstm_model = create_lstm_model(len(tokenizer.word_index) + 1, max_len)
lstm_model.fit(x_train_padded, y_train, validation_data=(x_val_padded, y_val), epochs=5, batch_size=32)

# Fine-tune BERT and RoBERTa
def fine_tune_transformer(model_name, train_encodings, val_encodings, train_labels, val_labels):
    transformer_model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
                              loss='sparse_categorical_crossentropy', 
                              metrics=['accuracy'])
    history = transformer_model.fit(train_encodings, train_labels, 
                                    validation_data=(val_encodings, val_labels), 
                                    epochs=3, batch_size=16)
    return transformer_model

# Tokenize for BERT and RoBERTa
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
bert_train_encodings = bert_tokenizer(x_train, truncation=True, padding=True, max_length=max_len, return_tensors="tf")
bert_val_encodings = bert_tokenizer(x_val, truncation=True, padding=True, max_length=max_len, return_tensors="tf")
roberta_train_encodings = roberta_tokenizer(x_train, truncation=True, padding=True, max_length=max_len, return_tensors="tf")
roberta_val_encodings = roberta_tokenizer(x_val, truncation=True, padding=True, max_length=max_len, return_tensors="tf")

# Train BERT and RoBERTa
bert_model = fine_tune_transformer('bert-base-uncased', bert_train_encodings, bert_val_encodings, y_train, y_val)
roberta_model = fine_tune_transformer('roberta-base', roberta_train_encodings, roberta_val_encodings, y_train, y_val)

# Evaluate models
print("LSTM Evaluation:")
lstm_results = lstm_model.evaluate(x_val_padded, y_val)

print("BERT Evaluation:")
bert_results = bert_model.evaluate(bert_val_encodings, y_val)

print("RoBERTa Evaluation:")
roberta_results = roberta_model.evaluate(roberta_val_encodings, y_val)

