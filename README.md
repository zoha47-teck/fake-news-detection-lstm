# fake-news-detection-lstm
This project uses an LSTM (Long Short-Term Memory) neural network to classify news articles as Real or Fake using deep learning and NLP techniques.

Features
Text preprocessing (cleaning, tokenization, stopword removal)
Word embeddings using TensorFlow/Keras Tokenizer
LSTM model for sequence classification
Training & evaluation on Fake/Real news datasets
Accuracy, loss, and confusion matrix included

Dataset
The project uses two CSV files:
fake.csv
true.csv
Both files contain news article text along with labels.

Model
The LSTM model includes:
Embedding layer
LSTM layer
Dense output layer with sigmoid activation
Optimized using binary crossentropy and Adam optimizer

How to Run
Install required libraries:
pip install tensorflow numpy pandas matplotlib scikit-learn wordcloud
Open the notebook:
fake_news_lstm.ipynb
Run all cells to:
preprocess the dataset
train the model
evaluate accuracy

Results
The notebook prints:
Training accuracy
Validation accuracy
Loss curve
Confusion matrix
Predictions samples

License
This project is for educational purposes only.
