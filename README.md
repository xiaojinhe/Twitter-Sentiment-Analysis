# Twitter-Sentiment-Analysis
- Built a classifier for inferring sentiment based on NLP methods and deep learning from ~1.4M tweets.
- Used BeautifulSoup and Regular Expressions in Python to remove features (e.g. Urls, hashtags) and translate
emoticons, handling negation, and then applied spelling correction, stemmer and tokenization on the dataset.
- Developed 4 models involving Embedding, CNN, Maxpooling, LSTM, Dense layers on both TensorFlow and
Keras, visualized the training results on TensorBoard with tf.summary and keras.callbacks.
- Improved a test accuracy from 76% to 84% and F1 score from 75% to 83% on a dataset consisting of 160, 000
tweets, after data preprocessing and tuning hyperparameters on the models.
