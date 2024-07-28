import streamlit as st
import pickle
from textblob import TextBlob
import numpy as np

# Load TF-IDF model
with open(r'/Users/kehin/PythonProjects/MRP/FINAL/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_model = pickle.load(f)

def predict_tfidf(text):
    # Transform the input text and return the TF-IDF features
    tfidf_features = tfidf_model.transform([text])
    return tfidf_features.toarray()

def get_sentiment(text):
    # Categorize sentiment
    analysis = TextBlob(text)
    if analysis.sentiment.polarity >= 0.05:
        return 'positive',analysis.sentiment.polarity
    elif analysis.sentiment.polarity <= -0.05:
        return 'negative',analysis.sentiment.polarity
    else:
        return 'neutral',analysis.sentiment.polarity

def main():
    st.title('Tweet Analysis App')

    # Input text from user
    user_input = st.text_area('Enter your tweet:', '')

    if user_input:
        # # Predict TF-IDF
        # tfidf_features = predict_tfidf(user_input)
        # st.write('TF-IDF Features:')
        # st.write(tfidf_features)

        # Predict Sentiment
        sentiment,score = get_sentiment(user_input)
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Sentiment Score: {score}')

if __name__ == '__main__':
    main()
