from flask import render_template, request, jsonify
from model.bert_model import load_bert_model, predict_answer

# Load the model, tokenizer, and device
model, tokenizer, device = load_bert_model()

def setup_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/analyze', methods=['POST'])
    def analyze():
        query = request.form.get('query', "")
        if not query.strip():
            return jsonify({'error': 'No input provided'})

        # Predict using BERT
        answer = predict_answer(query, model, tokenizer, device)

        return render_template('results.html', answer=answer, question=query)
