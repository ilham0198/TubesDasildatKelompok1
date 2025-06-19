import streamlit as st
import joblib
import numpy as np
from tensorflow import keras
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Tambahan untuk cleaning identik dengan Colab
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resource NLTK jika belum ada
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load models
@st.cache_resource
def load_dt_model():
    return joblib.load('Model/model_sentiment_dt.joblib')

@st.cache_resource
def load_nn_model():
    return keras.models.load_model('Model/model_sentiment_nn (1).h5')

@st.cache_resource
def load_regresi_model():
    return keras.models.load_model('Model/model_regresi_nn.keras')

@st.cache_resource
def load_vectorizer():
    return joblib.load('tfidf_vectorizer_decisiontree.joblib')

@st.cache_resource
def load_tokenizer_klasifikasi():
    with open('tokenizer_sentiment_klasifikasi.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_tokenizer_regresi():
    with open('tokenizer_sentiment_regresi.pkl', 'rb') as f:
        return pickle.load(f)

# Sentiment prediction (classification)
def predict_sentiment(text, model, tokenizer, maxlen=200, threshold=0.4):
    seq = tokenizer.texts_to_sequences([text])
    pad = keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(pad)
    return 'positive' if pred[0][0] > threshold else 'negative'

# Regression prediction
def predict_regression(text, model, tokenizer, maxlen=200):
    seq = tokenizer.texts_to_sequences([text])
    pad = keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(pad)
    return float(pred[0][0])

# Text cleaning function
# Cleaning identik dengan Colab: lowercase, hapus HTML, hapus tanda baca, stopword removal, lemmatization

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)  # hapus tag HTML
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # hapus tanda baca/angka
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    # Jangan hapus kata negasi penting
    negation_words = {'not', "n't", 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'hardly', 'scarcely', 'barely', 'nor'}
    tokens = [lemmatizer.lemmatize(w) for w in tokens if (w not in stop_words or w in negation_words)]
    return ' '.join(tokens)

# Load tokenizer (from Colab, user must provide tokenizer config if needed)
# For now, create a new one and fit on some example data if available
# In production, load the tokenizer from file (joblib/pickle/json)
tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000, oov_token='<OOV>')

st.title('Movie Review Sentiment Analysis Dashboard')

menu = st.sidebar.radio('Choose Task:', ['Classification', 'Regression'])

if menu == 'Classification':
    st.header('Classification')
    text = st.text_area('Enter a movie review:')
    model_type = st.selectbox('Choose Model:', ['Decision Tree', 'Neural Network'])
    if st.button('Predict Sentiment'):
        if not text.strip():
            st.warning('Please enter a review.')
        else:
            cleaned_text = clean_text(text)
            if model_type == 'Decision Tree':
                model = load_dt_model()
                vectorizer = load_vectorizer()
                X = vectorizer.transform([cleaned_text])
                pred = model.predict(X)
                sentiment = 'positive' if pred[0] == 1 else 'negative'
                st.success(f'Sentiment: {sentiment}')
            else:
                model = load_nn_model()
                tokenizer = load_tokenizer_klasifikasi()
                sentiment = predict_sentiment(cleaned_text, model, tokenizer, threshold=0.4)
                st.success(f'Sentiment: {sentiment}')
    st.markdown('---')
    st.subheader('Batch Upload (CSV)')
    uploaded_file = st.file_uploader('Upload CSV file with a column named "review"', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error('CSV must have a column named "review".')
        else:
            model_type_batch = st.selectbox('Model for Batch:', ['Decision Tree', 'Neural Network'], key='batch_model')
            if st.button('Predict Batch Sentiment'):
                reviews = df['review'].astype(str).tolist()
                if model_type_batch == 'Decision Tree':
                    model = load_dt_model()
                    vectorizer = load_vectorizer()
                    X = vectorizer.transform([clean_text(r) for r in reviews])
                    preds = model.predict(X)
                    sentiments = ['positive' if p == 1 else 'negative' for p in preds]
                else:
                    model = load_nn_model()
                    tokenizer = load_tokenizer_klasifikasi()
                    cleaned_reviews = [clean_text(r) for r in reviews]
                    seqs = tokenizer.texts_to_sequences(cleaned_reviews)
                    pads = keras.preprocessing.sequence.pad_sequences(seqs, maxlen=200)
                    preds = model.predict(pads)
                    sentiments = ['positive' if p[0] > 0.4 else 'negative' for p in preds]
                df['predicted_sentiment'] = sentiments
                st.dataframe(df)
                # Analisis error jika ada label asli
                if 'sentiment' in df.columns:
                    wrong = df[df['sentiment'] != df['predicted_sentiment']]
                    st.markdown('**Review yang salah prediksi:**')
                    st.dataframe(wrong[['review','sentiment','predicted_sentiment']])
                    st.markdown(f'Jumlah salah prediksi: {len(wrong)} dari {len(df)}')
                # Visualisasi distribusi prediksi
                st.markdown('**Distribusi Prediksi:**')
                fig, ax = plt.subplots()
                sns.countplot(x='predicted_sentiment', data=df, ax=ax)
                ax.set_title('Distribusi Prediksi Sentimen')
                st.pyplot(fig)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button('Download Results as CSV', csv, 'sentiment_results.csv', 'text/csv')
elif menu == 'Regression':
    st.header('Regression')
    text = st.text_area('Enter a movie review:')
    if st.button('Predict Score'):
        if not text.strip():
            st.warning('Please enter a review.')
        else:
            model = load_regresi_model()
            tokenizer = load_tokenizer_regresi()
            score = predict_regression(text, model, tokenizer)
            st.success(f'Predicted Score: {score:.2f}')
    st.markdown('---')
    st.subheader('Batch Upload (CSV)')
    uploaded_file = st.file_uploader('Upload CSV file with a column named "review"', type=['csv'], key='reg_batch')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Ubah nama kolom 'Review' menjadi 'review' jika perlu
        if 'review' not in df.columns and 'Review' in df.columns:
            df.rename(columns={'Review': 'review'}, inplace=True)
        if 'review' not in df.columns:
            st.error('CSV must have a column named "review".')
        else:
            if st.button('Predict Batch Score'):
                reviews = df['review'].astype(str).tolist()
                model = load_regresi_model()
                tokenizer = load_tokenizer_regresi()
                seqs = tokenizer.texts_to_sequences(reviews)
                pads = keras.preprocessing.sequence.pad_sequences(seqs, maxlen=200)
                preds = model.predict(pads)
                scores = [float(p[0]) for p in preds]
                df['predicted_score'] = scores
                st.dataframe(df)
                # Analisis error jika ada label asli
                if 'score' in df.columns:
                    df['score'] = pd.to_numeric(df['score'], errors='coerce')
                    df['error'] = abs(df['score'] - df['predicted_score'])
                    worst = df.sort_values('error', ascending=False).head(10)
                    st.markdown('**10 Review dengan error prediksi terbesar:**')
                    st.dataframe(worst[['review','score','predicted_score','error']])
                    st.markdown(f'Rata-rata error: {df["error"].mean():.2f}')
                # Visualisasi distribusi prediksi
                st.markdown('**Distribusi Skor Prediksi:**')
                fig, ax = plt.subplots()
                sns.histplot(df['predicted_score'], bins=20, kde=True, ax=ax)
                ax.set_title('Distribusi Skor Prediksi')
                st.pyplot(fig)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button('Download Results as CSV', csv, 'regression_results.csv', 'text/csv')

st.markdown('---')
st.caption('Model files: model_sentiment_dt.joblib, model_sentiment_nn (1).h5, model_regresi_nn.keras')
