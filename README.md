# NLP Social Media Sentiment & Topic Analysis

A comprehensive Natural Language Processing project for analyzing sentiment and discovering topics in social media data. This project processes 31,962 tweets to extract insights about brand perception, customer sentiment, and emerging trends using advanced NLP techniques, machine learning models, and deep learning architectures.

## 📊 Dataset

- **Size**: 31,962 tweets
- **Features**: 6 features
- **Target Variable**: Sentiment (positive, negative, neutral)
- **Source**: Twitter Sentiment Analysis Dataset

## 🎯 Features

This project implements six major components:

1. **Data Preprocessing Pipeline** - Comprehensive text cleaning and feature extraction
2. **Exploratory Data Analysis** - Sentiment patterns, word frequency, and temporal trends
3. **Supervised Machine Learning** - Multiple classifiers for sentiment prediction
4. **Unsupervised Machine Learning** - Topic modeling and clustering for trend discovery
5. **Deep Learning Models** - Transformer-based and neural network architectures
6. **Streamlit Dashboard** - Real-time sentiment analysis and visualization interface

## 📋 Requirements

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended for deep learning models)
- **Storage**: At least 2GB free space for models and data

### Required Libraries

The project uses the following key libraries:

- **NLP & Text Processing**: NLTK, spaCy
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: PyTorch or TensorFlow, Transformers (Hugging Face)
- **Visualization**: Matplotlib, Seaborn, Plotly, pyLDAvis
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Utilities**: tqdm, joblib

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd NLP_Social-Media-Sentiment
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Step 5: Download Dataset

Place your dataset file (`twitter_sentiment.csv` or similar) in the `data/` directory.

## 📁 Project Structure

```
NLP_Social-Media-Sentiment/
│
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── processed/              # Cleaned and preprocessed data
│   └── models/                 # Saved model files
│
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py     # Text cleaning utilities
│   │   ├── tokenizer.py        # Tokenization with NLTK TweetTokenizer
│   │   ├── lemmatizer.py       # POS-tagged lemmatization
│   │   └── feature_extractor.py # Hashtag and mention extraction
│   │
│   ├── eda/
│   │   ├── sentiment_analysis.py    # Sentiment distribution analysis
│   │   ├── word_frequency.py         # Word frequency and n-grams
│   │   ├── network_analysis.py      # Retweet and mention networks
│   │   ├── temporal_analysis.py      # Time series sentiment trends
│   │   └── geographic_mapping.py     # Geographic sentiment visualization
│   │
│   ├── supervised_ml/
│   │   ├── baseline_model.py        # TF-IDF + Logistic Regression
│   │   ├── random_forest.py         # Random Forest with n-grams
│   │   ├── svm_classifier.py        # SVM with linear kernel
│   │   └── evaluation.py            # Cross-validation and metrics
│   │
│   ├── unsupervised_ml/
│   │   ├── lda_modeling.py          # Latent Dirichlet Allocation
│   │   ├── kmeans_clustering.py     # KMeans on TF-IDF vectors
│   │   ├── nmf_decomposition.py     # Non-negative Matrix Factorization
│   │   └── topic_visualization.py   # pyLDAvis interactive visualization
│   │
│   ├── deep_learning/
│   │   ├── distilbert_finetune.py   # Fine-tuned DistilBERT
│   │   ├── lstm_model.py            # LSTM with embeddings
│   │   ├── cnn_classifier.py        # CNN with multiple filter sizes
│   │   └── attention_mechanism.py   # Attention-based models
│   │
│   └── streamlit_app/
│       ├── app.py                   # Main Streamlit application
│       ├── components/
│       │   ├── sentiment_input.py   # Real-time text input
│       │   ├── topic_display.py     # Topic assignment display
│       │   ├── trend_visualization.py # Plotly trend charts
│       │   ├── alert_system.py      # Negative sentiment alerts
│       │   └── export_tools.py     # Report export functionality
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_supervised_ml.ipynb
│   ├── 04_unsupervised_ml.ipynb
│   └── 05_deep_learning.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 Usage Guide

### Data Preprocessing

```python
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tokenizer import TweetTokenizer
from src.preprocessing.lemmatizer import POSLemmatizer
from src.preprocessing.feature_extractor import FeatureExtractor

# Initialize components
cleaner = TextCleaner()
tokenizer = TweetTokenizer()
lemmatizer = POSLemmatizer()
extractor = FeatureExtractor()

# Process tweets
cleaned_text = cleaner.clean(tweet)
tokens = tokenizer.tokenize(cleaned_text)
lemmatized = lemmatizer.lemmatize(tokens)
hashtags, mentions = extractor.extract_features(tweet)
```

### Exploratory Data Analysis

```bash
# Run EDA notebook
jupyter notebook notebooks/02_exploratory_analysis.ipynb

# Or run Python script
python src/eda/sentiment_analysis.py
```

### Supervised Machine Learning

```python
from src.supervised_ml.baseline_model import BaselineModel
from src.supervised_ml.random_forest import RandomForestClassifier
from src.supervised_ml.svm_classifier import SVMClassifier

# Train baseline model
baseline = BaselineModel()
baseline.train(X_train, y_train)
baseline.evaluate(X_test, y_test)

# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.train(X_train, y_train)

# Train SVM
svm_model = SVMClassifier()
svm_model.train(X_train, y_train)
```

### Unsupervised Machine Learning

```python
from src.unsupervised_ml.lda_modeling import LDAModel
from src.unsupervised_ml.kmeans_clustering import KMeansClustering
from src.unsupervised_ml.nmf_decomposition import NMFModel

# LDA Topic Modeling
lda = LDAModel(n_topics=15)
lda.fit(tfidf_vectors)
topics = lda.get_topics()

# KMeans Clustering
kmeans = KMeansClustering(n_clusters=10)
clusters = kmeans.fit_predict(tfidf_vectors)

# NMF Decomposition
nmf = NMFModel(n_components=15)
nmf.fit(tfidf_vectors)
```

### Deep Learning

```python
from src.deep_learning.distilbert_finetune import DistilBERTFineTuner
from src.deep_learning.lstm_model import LSTMSentimentClassifier
from src.deep_learning.cnn_classifier import CNNSentimentClassifier

# Fine-tune DistilBERT
bert_model = DistilBERTFineTuner()
bert_model.train(train_loader, val_loader)
predictions = bert_model.predict(test_data)

# Train LSTM
lstm_model = LSTMSentimentClassifier()
lstm_model.train(X_train, y_train)

# Train CNN
cnn_model = CNNSentimentClassifier()
cnn_model.train(X_train, y_train)
```

### Streamlit Dashboard

```bash
# Launch the Streamlit app
streamlit run src/streamlit_app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📖 Detailed Feature Documentation

### 1. Data Preprocessing Pipeline

**Purpose**: Clean and normalize tweet text for downstream analysis and modeling.

**Key Techniques**:
- **Text Cleaning**: Removes URLs, @mentions, #hashtags, and special characters while preserving important information
- **Tokenization**: Uses NLTK TweetTokenizer optimized for social media text
- **Lemmatization**: Context-aware normalization using POS tagging for accurate word form reduction
- **Feature Extraction**: Separates hashtags and mentions as distinct features for analysis
- **Vectorization**: TF-IDF with sublinear term frequency scaling for better feature representation

**Implementation Details**:
- URL removal using regex patterns
- Hashtag and mention preservation in separate columns
- POS-tagged lemmatization using WordNet
- TF-IDF vectorization with min_df=2, max_df=0.95

**Output**: Cleaned text, tokenized sequences, lemmatized tokens, extracted features, and TF-IDF vectors

### 2. Exploratory Data Analysis

**Purpose**: Understand sentiment patterns, identify brand perception drivers, and discover insights.

**Key Techniques**:
- **Sentiment Distribution**: Analysis over time and by entity/brand
- **Word Frequency Analysis**: Most common words and n-grams for each sentiment class
- **Network Analysis**: Graph-based visualization of retweet and mention patterns
- **Temporal Trends**: Time series analysis with anomaly detection for sentiment shifts
- **Geographic Mapping**: Sentiment visualization by location (where available)

**Implementation Details**:
- Time series decomposition for trend analysis
- NetworkX for graph construction and analysis
- Statistical tests for anomaly detection
- Interactive maps using Plotly or Folium

**Output**: Visualizations, statistical summaries, network graphs, and trend reports

### 3. Supervised Machine Learning

**Purpose**: Build high-accuracy sentiment classifiers for real-time brand reputation monitoring.

**Key Techniques**:
- **Baseline Model**: TF-IDF + Logistic Regression with regularization
- **Random Forest**: Ensemble method with n-gram features (unigrams and bigrams)
- **SVM**: Linear kernel optimized for high-dimensional text data
- **Evaluation**: Stratified k-fold cross-validation to handle class imbalance
- **Sampling**: SMOTE or class weights for imbalanced datasets

**Implementation Details**:
- Grid search for hyperparameter tuning
- Stratified sampling for train/test splits
- Class balancing using SMOTE or weighted loss functions
- Feature engineering with n-gram ranges (1,2)

**Output**: Trained models, performance metrics (accuracy, precision, recall, F1-score), confusion matrices

### 4. Unsupervised Machine Learning

**Purpose**: Discover emerging topics and trends automatically without labeled data.

**Key Techniques**:
- **LDA (Latent Dirichlet Allocation)**: Probabilistic topic modeling with 15-20 topics
- **KMeans Clustering**: Partitioning tweets into k=10 clusters based on TF-IDF similarity
- **NMF (Non-negative Matrix Factorization)**: Additive topic decomposition for interpretable topics
- **Interactive Visualization**: pyLDAvis for topic exploration and validation
- **Quality Assessment**: Human-in-the-loop validation for topic coherence

**Implementation Details**:
- Coherence score calculation for topic quality
- Optimal topic number selection using perplexity and coherence
- Topic-word and document-topic distributions
- Interactive dashboards for topic exploration

**Output**: Topic models, cluster assignments, topic visualizations, topic-word distributions

### 5. Deep Learning Models

**Purpose**: Achieve state-of-the-art sentiment classification using advanced neural architectures.

**Key Techniques**:
- **DistilBERT Fine-tuning**: Efficient transformer model for sentiment classification
- **LSTM Networks**: Bidirectional LSTM with embedding layers for sequence modeling
- **CNN Classifier**: Multi-filter CNN (3, 4, 5-gram filters) for sentence classification
- **Attention Mechanisms**: Interpretable attention weights for prediction explanation
- **Performance Comparison**: Benchmark against traditional ML methods

**Implementation Details**:
- Transfer learning from pre-trained DistilBERT
- Word embeddings (Word2Vec, GloVe, or learned embeddings)
- Multi-head attention for feature extraction
- Early stopping and learning rate scheduling

**Output**: Fine-tuned models, training curves, attention visualizations, performance benchmarks

### 6. Streamlit Dashboard

**Purpose**: Real-time sentiment analysis interface for social media managers.

**Key Features**:
- **Real-time Text Input**: Instant sentiment prediction for user-entered text
- **Topic Assignment**: Automatic topic labeling for input text
- **Similar Content Display**: Show tweets with similar sentiment/topics
- **Sentiment Trend Visualization**: Interactive Plotly charts showing sentiment over time
- **Alert System**: Notifications for negative sentiment spikes or anomalies
- **Export Capabilities**: Generate PDF/CSV reports for social media analysis

**Implementation Details**:
- Real-time model inference using cached models
- Cosine similarity for finding similar content
- Plotly for interactive visualizations
- Background job processing for batch analysis
- Report generation with charts and statistics

**Output**: Interactive web dashboard, real-time predictions, trend charts, exportable reports

## 📈 Results & Performance

### Model Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression (Baseline) | 0.XX | 0.XX | 0.XX | 0.XX |
| Random Forest | 0.XX | 0.XX | 0.XX | 0.XX |
| SVM (Linear) | 0.XX | 0.XX | 0.XX | 0.XX |
| DistilBERT (Fine-tuned) | 0.XX | 0.XX | 0.XX | 0.XX |
| LSTM | 0.XX | 0.XX | 0.XX | 0.XX |
| CNN | 0.XX | 0.XX | 0.XX | 0.XX |

*Note: Replace XX with actual performance metrics after model training*

### Key Insights

- **Sentiment Distribution**: [Add insights about sentiment distribution]
- **Top Topics**: [List discovered topics]
- **Temporal Patterns**: [Describe time-based trends]
- **Geographic Insights**: [Location-based findings]

## 🔧 Configuration

Create a `config.yaml` file to customize model parameters:

```yaml
preprocessing:
  min_df: 2
  max_df: 0.95
  ngram_range: [1, 2]
  
models:
  lda:
    n_topics: 15
  kmeans:
    n_clusters: 10
  random_forest:
    n_estimators: 100
    max_depth: 20
  
deep_learning:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 3
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Add docstrings to all functions and classes
- Include type hints where applicable
- Write unit tests for new features

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Twitter Sentiment Analysis Dataset
- **Libraries**: 
  - NLTK for natural language processing
  - scikit-learn for machine learning
  - Hugging Face Transformers for deep learning models
  - Streamlit for web application framework
  - Plotly for interactive visualizations
  - pyLDAvis for topic model visualization