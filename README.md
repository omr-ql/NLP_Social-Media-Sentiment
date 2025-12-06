# NLP Social Media Sentiment & Topic Analysis

A comprehensive Natural Language Processing project for analyzing sentiment and discovering topics in social media data. This project processes 31,962 tweets to extract insights about brand perception, customer sentiment, and emerging trends using advanced NLP techniques, machine learning models, and deep learning architectures.

## 📊 Dataset

- **Size**: 31,962 tweets
- **Features**: 6 features
- **Target Variable**: Sentiment (positive, negative, neutral, irrelevant)
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

- **NLP & Text Processing**: NLTK, Gensim
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Deep Learning**: PyTorch, TensorFlow, Transformers (Hugging Face)
- **Visualization**: Matplotlib, Seaborn, pyLDAvis
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Utilities**: tqdm, joblib

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/omr-ql/NLP_Social-Media-Sentiment.git
cd NLP_Social-Media-Sentiment

# Install dependencies
pip install pandas numpy matplotlib seaborn nltk scikit-learn imbalanced-learn streamlit transformers torch gensim pyLDAvis joblib

# Run the Streamlit dashboard
streamlit run app.py

# Or open the complete project notebook
jupyter notebook Complete_project.ipynb
```

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/omr-ql/NLP_Social-Media-Sentiment.git
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

If `requirements.txt` is not available, install the core dependencies:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn imbalanced-learn streamlit transformers torch tensorflow gensim pyLDAvis joblib
```

For PyTorch with CUDA support (optional):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
```

**Note**: The notebooks will automatically download these resources if they're not already present.

### Step 5: Verify Dataset

The dataset files (`twitter_training.csv` and `twitter_validation.csv`) are already included in the repository.

## 📁 Project Structure

```
NLP_Social-Media-Sentiment/
│
├── app.py                          # Streamlit dashboard application
│
├── notebooks/
│   ├── Complete_project.ipynb      # Complete end-to-end project notebook
│   ├── final_project.ipynb         # Final project implementation
│   └── unsupervized-final (2).ipynb # Unsupervised learning notebook
│
├── models/
│   ├── best_sentiment_classifier.pkl    # Best trained sentiment classifier
│   ├── gensim_lda_model.pkl            # Gensim LDA topic model
│   └── gensim_dictionary.pkl           # Gensim dictionary for topic modeling
│
├── data/
│   ├── twitter_training.csv        # Training dataset (31,962 tweets)
│   └── twitter_validation.csv      # Validation dataset
│
├── visualizations/                 # Generated visualization files
│   ├── sentiment_distribution.png
│   ├── sentiment_by_entity.png
│   ├── entity_distribution.png
│   ├── tweet_length_distribution.png
│   ├── top_mentions.png
│   ├── top_1_1grams_*.png          # Word frequency by sentiment
│   └── top_2_2grams_*.png          # Bigram frequency by sentiment
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 Usage Guide

### Running the Complete Project Notebook

The main project implementation is in `Complete_project.ipynb`, which contains all the analysis from preprocessing to model training:

```bash
# Launch Jupyter Notebook
jupyter notebook Complete_project.ipynb
```

This notebook includes:
- Data preprocessing with NLTK TweetTokenizer and POS-tagged lemmatization
- Exploratory data analysis with visualizations
- Supervised ML models (Logistic Regression, Random Forest, SVM)
- Unsupervised ML (LDA, KMeans, NMF)
- Deep Learning models (DistilBERT, LSTM, CNN)

### Running Individual Notebooks

```bash
# Final project notebook (supervised learning focus)
jupyter notebook final_project.ipynb

# Unsupervised learning notebook (topic modeling)
jupyter notebook unsupervized-final\ \(2\).ipynb
```

### Streamlit Dashboard

Launch the interactive sentiment analysis dashboard:

```bash
# Make sure you have the model files in the root directory:
# - distilbert_sentiment_classifier.pt (or best_sentiment_classifier.pkl)
# - gensim_lda_model.pkl
# - gensim_dictionary.pkl

streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Features:**
- Real-time sentiment prediction for tweet text
- Topic assignment using LDA model
- Interactive text input and analysis

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
- **Real-time Text Input**: Instant sentiment prediction for user-entered text using fine-tuned DistilBERT
- **Topic Assignment**: Automatic topic labeling using Gensim LDA model
- **Preprocessing Pipeline**: Integrated text cleaning, tokenization, and lemmatization
- **Multi-class Classification**: Supports 4 sentiment classes (Positive, Negative, Neutral, Irrelevant)

**Implementation Details**:
- Fine-tuned DistilBERT model for sentiment classification
- Gensim LDA model for topic modeling (15-20 topics)
- Real-time preprocessing with NLTK TweetTokenizer
- POS-tagged lemmatization for accurate text normalization
- Model caching for fast inference

**Output**: Interactive web dashboard with real-time sentiment and topic predictions

## 📈 Results & Performance

### Model Performance Metrics

Performance metrics are available in the `Complete_project.ipynb` notebook. The project includes:

- **Baseline Models**: TF-IDF + Logistic Regression, Random Forest, SVM
- **Deep Learning Models**: Fine-tuned DistilBERT, LSTM, CNN
- **Evaluation**: Stratified cross-validation with class imbalance handling

### Generated Visualizations

The project generates several visualization files:

- `sentiment_distribution.png` - Overall sentiment distribution
- `sentiment_by_entity.png` - Sentiment breakdown by entity/brand
- `entity_distribution.png` - Entity frequency analysis
- `tweet_length_distribution.png` - Tweet length statistics
- `top_mentions.png` - Most mentioned entities
- `top_1_1grams_*.png` - Top unigrams for each sentiment class
- `top_2_2grams_*.png` - Top bigrams for each sentiment class

### Saved Models

- `best_sentiment_classifier.pkl` - Best performing sentiment classifier
- `gensim_lda_model.pkl` - Trained LDA topic model (15-20 topics)
- `gensim_dictionary.pkl` - Gensim dictionary for topic modeling

## 🔧 Model Files

The project uses the following pre-trained models:

- **Sentiment Classifier**: `best_sentiment_classifier.pkl` or `distilbert_sentiment_classifier.pt`
- **Topic Model**: `gensim_lda_model.pkl` (Gensim LDA with 15-20 topics)
- **Dictionary**: `gensim_dictionary.pkl` (Vocabulary for topic modeling)

**Note**: For the Streamlit app to work, ensure these model files are in the root directory. The DistilBERT model file (`distilbert_sentiment_classifier.pt`) should be present for the app to load the fine-tuned transformer model.

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
  - NLTK for natural language processing and text preprocessing
  - scikit-learn for machine learning models and evaluation
  - Gensim for topic modeling (LDA)
  - Hugging Face Transformers for DistilBERT fine-tuning
  - PyTorch and TensorFlow for deep learning models
  - Streamlit for web application framework
  - pyLDAvis for interactive topic model visualization
  - imbalanced-learn for handling class imbalance