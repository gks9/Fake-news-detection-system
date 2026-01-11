from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and vectorizer
print("Loading model and vectorizer...")
tfidf = joblib.load('tfidf.pkl')
clf = joblib.load('logreg.pkl')
print("✓ Model loaded successfully")

# Download stopwords if needed
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

def clean_text(text):
    """Clean text same way as training"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/')
def home():
    return jsonify({
        'message': 'Fake News Detector API',
        'endpoints': {
            '/predict': 'POST - Send {"text": "your news text"} to get prediction'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in request body'
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Clean and transform
        cleaned = clean_text(text)
        vec = tfidf.transform([cleaned])
        
        # Predict
        label = clf.predict(vec)[0]
        proba = clf.predict_proba(vec)[0]
        confidence = float(proba.max())
        
        # Return result
        result = {
            'label': 'Fake' if label == 1 else 'Real',
            'confidence': confidence,
            'text_preview': text[:100] + '...' if len(text) > 100 else text
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
