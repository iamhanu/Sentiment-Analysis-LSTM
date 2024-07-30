import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == "positive" else 0)
    reviews = df['review'].values
    labels = df['sentiment'].values
    return reviews, labels

def preprocess_data(reviews, max_words=10000, max_len=200):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    data = pad_sequences(sequences, maxlen=max_len)
    return data, tokenizer
