from flask import Flask, request, jsonify
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

app = Flask(__name__)

# Load the pickled model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    if pd.isnull(text):
        return ''

    # Tokenization
    tokens = word_tokenize(text.lower())

    # Removing stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    filtered_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Combine tokens into a single string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

@app.route('/', methods=['GET'])
def mainfn():
    return jsonify({'prediction': "true"})

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON input data
    data = request.get_json()
    print(data["data"])
    processed_reviews = [preprocess_text(review) for review in data["data"]]
    X_sample_tfidf = tfidf_vectorizer.transform(processed_reviews)

    try:
        # Make prediction using the model
        prediction = model.predict(X_sample_tfidf)[0]
        # Assuming model.predict returns a single prediction, adjust if necessary
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
