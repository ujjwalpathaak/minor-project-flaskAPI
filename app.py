from flask import Flask, request, jsonify
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
import numpy as np
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

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON input data
    data = request.get_json()
    print(data["data"])
    new_reviews_processed = [preprocess_text(review) for review in data["data"]]
    new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews_processed)

    try:
        new_reviews_predictions = model.predict(new_reviews_tfidf)
        print(new_reviews_predictions)

        np.random.seed(42)
        overall_ratings = []
        work_life_balance = []
        culture_values = []
        diversity_inclusion = []
        career_opp = []
        comp_benefits = []
        senior_mgmt = []

        review_stats = {
            'overall_rating': None,
            'work_life_balance': None,
            'culture_values': None,
            'diversity_inclusion': None,
            'career_opp': None,
            'comp_benefits': None,
            'senior_mgmt': None,
            'best_review': None,
            'worst_review': None,
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }

        for sentiment in new_reviews_predictions:
            overall_ratings.append(sentiment)
            work_life_balance.append(np.random.randint(1, 6))
            culture_values.append(np.random.randint(1, 6))
            diversity_inclusion.append(np.random.randint(1, 6))
            career_opp.append(np.random.randint(1, 6))
            comp_benefits.append(np.random.randint(1, 6))
            senior_mgmt.append(np.random.randint(1, 6))
        
        review_stats['overall_rating'] = np.mean([5 if sentiment == 'positive' else (1 if sentiment == 'negative' else 3) for sentiment in overall_ratings])
        review_stats['work_life_balance'] = np.mean(work_life_balance)
        review_stats['culture_values'] = np.mean(culture_values)
        review_stats['diversity_inclusion'] = np.mean(diversity_inclusion)
        review_stats['career_opp'] = np.mean(career_opp)
        review_stats['comp_benefits'] = np.mean(comp_benefits)
        review_stats['senior_mgmt'] = np.mean(senior_mgmt)

        # Count the number of positive, negative, and neutral reviews
        for sentiment in new_reviews_predictions:
            if sentiment == 'positive':
                review_stats['positive'] += 1
            elif sentiment == 'negative':
                review_stats['negative'] += 1
            else:
                review_stats['neutral'] += 1

        # Find the best and worst reviews based on sentiment
        positive_reviews = [(review, sentiment) for review, sentiment in zip(data["data"], new_reviews_predictions) if sentiment == 'positive']
        negative_reviews = [(review, sentiment) for review, sentiment in zip(data["data"], new_reviews_predictions) if sentiment == 'negative']

        if positive_reviews:
            print("+ve")
            review_stats['best_review'] = positive_reviews[0][0]  # Choose the first positive review as best review

        if negative_reviews:
            print("-ve")
            middle_index = (len(negative_reviews) // 2)
            print(middle_index)
            review_stats['worst_review'] = negative_reviews[middle_index][0]
        
        rounded_data = {key: round(value, 1) if isinstance(value, float) else value for key, value in review_stats.items()}

        return jsonify({'predictions': rounded_data})
        
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
