from flask import Flask, request, render_template
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model('app/sentiment_analysis_model.h5')
with open('app/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 200

def preprocess_review(review):
    sequence = tokenizer.texts_to_sequences([review])
    data = pad_sequences(sequence, maxlen=max_len)
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    data = preprocess_review(review)
    prediction = model.predict(data)[0][0]
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
