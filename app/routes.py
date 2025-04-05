from flask import render_template, request, jsonify

#from model.bart_model import load_bart_model,predict_bart_sentiment
from model.bert_model import load_bert_model, predict_sentiment
from model.preprocessor import data_augmentation;

# Load the CNN model and tokenizer
cnn_model, tokenizer = load_cnn_model()
#bart_model, tokenizer = load_bart_model()

def setup_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    """@app.route('/analyze', methods=['POST'])
    def analyze():
        query = request.form.get('query', "")
        if not query.strip():
            return jsonify({'error': 'No input provided'})

        # Predict sentiment
        sentiment = predict_sentiment(query, model, tokenizer)

        # Map sentiment to a label
        sentiment_map = {0: "Negative", 1: "Positive"}
        sentiment_label = sentiment_map[sentiment]

        return render_template('results.html', sentiment=sentiment_label, tweet=query)"""

    @app.route('/analyze', methods=['POST'])
    def analyze():
        query = request.form.get('query', "")
        model_selection = request.form.get('model', "")
        print("Selected Model:", model_selection)
        if not query.strip():
            return jsonify({'error': 'No input provided'})
        sentiment = '';
        if model_selection == 'lstm':
            print("Inside LSTM")
            sentiment = predict_bert_sentiment(query, lstm_model, tokenizer)

        else:
            sentiment = predict_sentiment(query, cnn_model, tokenizer)
        # Predict sentiment
        print("sentiment::::", sentiment)
        # Map sentiment to a label
        sentiment_map = {0: "Very Negative", 1: "Negative", 2 : "Nutral", 2: " Positive" , 3: " Very Positive"}
        sentiment_label = sentiment_map[sentiment]

        return render_template('results.html', sentiment=sentiment_label, tweet=query)

    @app.route('/data-augmentation')
    def display_data_augmentation():
        # Get data augmentation results
        results = data_augmentation()

        # Pass results to the template
        return render_template('data_augmentation.html', results=results)
