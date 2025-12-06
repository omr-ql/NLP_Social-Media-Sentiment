import streamlit as st
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib
import gensim
from gensim.corpora import Dictionary
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# NLTK Setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Preprocessing
tokenizer_nltk = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def preprocess_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned = ' '.join(text.split())
    if not cleaned: return ""
    tokens = tokenizer_nltk.tokenize(cleaned.lower())
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in pos_tag(tokens) if w not in stop_words and len(w)>2]
    return ' '.join(lemmatized)

# Load Models
sentiment_to_int = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}
int_to_sentiment = {v: k for k, v in sentiment_to_int.items()}

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
try:
    model.load_state_dict(torch.load('distilbert_sentiment_classifier.pt', map_location=torch.device('cpu')))
except:
    st.error("Model file not found.")
model.eval()

try:
    gensim_lda_model = joblib.load('gensim_lda_model.pkl')
    gensim_dictionary = joblib.load('gensim_dictionary.pkl')
except:
    st.error("Topic models not found.")

def predict_sentiment(text):
    processed = preprocess_tweet(text)
    if not processed: return "N/A"
    inputs = tokenizer(processed, truncation=True, padding='max_length', return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return int_to_sentiment[pred]

def predict_topic(text):
    processed = preprocess_tweet(text)
    if not processed: return "N/A"
    bow = gensim_dictionary.doc2bow(processed.split())
    topics = gensim_lda_model.get_document_topics(bow)
    if not topics: return "No Topic"
    dom_topic = max(topics, key=lambda x: x[1])
    words = ", ".join([w for w, p in gensim_lda_model.show_topic(dom_topic[0], topn=5)])
    return f"Topic {dom_topic[0]} ({dom_topic[1]*100:.1f}%): {words}"

# UI
st.title('Tweet Sentiment & Topic Analyzer')
user_input = st.text_area("Enter tweet text here:")
if st.button("Analyze"):
    if user_input:
        with st.spinner('Analyzing...'):
            sent = predict_sentiment(user_input)
            topic = predict_topic(user_input)
            st.success(f"Sentiment: {sent}")
            st.info(f"Topic: {topic}")
    else:
        st.warning("Please enter text.")
