import re
import joblib
from flask import Flask, request, jsonify

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON: {"message": "Your text here"}
    Returns: {"prediction": "SPAM" or "HAM", "spam_probability": 0.98}
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field'}), 400

    raw_message = data['message']
    cleaned = clean_text(raw_message)
    vec = vectorizer.transform([cleaned])
    
    prob_spam = model.predict_proba(vec)[0][1]   # probability of spam
    prediction = "SPAM" if prob_spam >= 0.5 else "HAM"
    
    return jsonify({
        'prediction': prediction,
        'spam_probability': round(prob_spam, 4)
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Spam Detection API is running. Use POST /predict with {"message": "..."}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)