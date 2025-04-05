
# ChemLLM App

This project is a web-based ChemLLM Question/Answer based application that uses machine learning models (BERT,  RoBERTa, MISTRAL-7B, LLAMA , TBD) to predict naswer for the give question. It is implemented using Python, TensorFlow/Pytorch, and Flask, with options for model training, serving, and testing.
## Feature 

Train ChemLLM models (BERT,  RoBERTa, MISTRAL-7B, LLAMA , TBD).

A Flask-based web interface for predicting sentiment.


### BERT 
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model pre-trained on a large corpus. It understands context by analyzing words in both directions (left-to-right and right-to-left). The BERT model in this project is fine-tuned for chemistroy question/answer based classification using a binary classification head.
### Rboberta 
RoBERTa (A Robustly Optimized BERT Pretraining Approach) is an enhanced version of BERT with improvements in training techniques and hyperparameters. It delivers higher accuracy for chemistroy question/answer tasks by leveraging more robust pretraining methods.

### MISTARL-7B 
TBD

# Falsk UI 
The application includes a Flask-based web interface for users to interact with the sentiment analysis models. Users can:

* Input text in the provided text field.

* Select the desired model (BERT,  RoBERTa, MISTRAL-7B, LLAMA , TBD) for sentiment prediction.

* View the predicted answer  and accuracy (positive or negative) instantly on the UI.

The Flask UI is designed to be intuitive and responsive, ensuring a seamless user experience. It integrates the trained models and exposes their prediction capabilities through RESTful API endpoints.
# Installation Instructions
## Clone the project
git clone https://github.com/rsdtinfo01/ChemLLM.git
## Setup Python environment
cd Chem-LLM-app
python -m venv 
venv venv\Scripts\activate 
pip install -r requirements.txt

## Train the model
Train the models (LSTM, BERT, and RoBERTa) using the provided scripts:

python model/bert_model.py
python model/roberta_model.py
python model/mistral7b_model.py
python model/llma_model.py

These scripts will save the trained models in the model/saved_models directory.


## Run the Flask app

### Start the Flask Web Application

python app/main.py

The app will be available at http://localhost:5000.

## Dockerize and run 
docker build -t chemllm-app .
docker run -d -p 5000:5000 chemllm-app
