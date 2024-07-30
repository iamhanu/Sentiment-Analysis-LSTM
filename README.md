# Movie-Review-Sentiment-Analysis
This is the sentiment analysis application for movie reviews using the IMDB dataset from kaggle.In this project, I developed a sentiment analysis application for movie reviews using the IMDB dataset. I utilized Python for data preprocessing and built an LSTM model with TensorFlow, achieving accurate classification of movie reviews as positive or negative. The model included an embedding layer, spatial dropout, and LSTM layers, followed by a dense output layer with a sigmoid activation function. I integrated this model into a Flask-based web application, enabling users to input reviews and receive real-time sentiment analysis.

There is no code in __init__.py and model.py files, __pycache__,  folder is created during the runtime.

# Steps to run
1) Run sentiment-analysis.ipynb file which will create the sentiment_analysis_model.h5 and tokenizer.pkl files
2) Then change your directory to the /sentiment-analysis folder and run the command "python app/app.py"
3) This will run the app at http://127.0.0.1:5000
