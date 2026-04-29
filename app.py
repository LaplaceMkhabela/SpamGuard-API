import re
import os
import json
import joblib
from flask import Flask, request, jsonify,render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

# --- Limiter Setup ---
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://redis_db:6379",
    strategy="fixed-window" 
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Profanity Filter
def check_profanity(text):
    with open(os.path.join("data", "bad_words.json"), "r") as file:
        data = json.load(file)
        profane_words = data['words']
        tokens = set(text.lower().split())
        return not tokens.isdisjoint(profane_words)
    
# Phishing Heuristic
def estimate_phishing(text):
    text = text.lower()
    with open(os.path.join("data", "phishing_data.json"), "r") as file:
        data = json.load(file)
        keywords = data['words']
        
        patterns = [r'http[s]?://', r'bit\.ly', r'click here', r'urgent']
    
        score = 0
        for word in keywords:
            if word in text: score += 0.15
        for pattern in patterns:
            if re.search(pattern, text): score += 0.25
            
        return min(score, 1.0)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field'}), 400

    raw_message = data['message']
    cleaned = clean_text(raw_message)
    
    # ML Inference for Spam
    vec = vectorizer.transform([cleaned])
    prob_spam = model.predict_proba(vec)[0][1]
    prediction = "SPAM" if prob_spam >= 0.5 else "HAM"

    # Heuristic Checks
    phishing_prob = estimate_phishing(raw_message)
    profanity_found = check_profanity(raw_message)
    
    return jsonify({
        'prediction': prediction,
        'spam_probability': round(prob_spam, 4),
        'phishing_probability': round(phishing_prob, 4),
        'profanity_detected': profanity_found,
        'language': 'en-US'
    })

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Custom error handler for when the limit is reached
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description)
    }), 429

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)