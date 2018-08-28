import pandas as pd
import numpy as np
import os
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dense
from keras import regularizers
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from keras.utils import plot_model
from data_loading import load_data_and_shuffle
import data_preprocess

raw_data_file = "./twitter-sentiment/data/raw_data_file.csv"
cleaned_data_dir = "./twitter-sentiment/data/"
cleaned_pos_file = "./twitter-sentiment/data/cleaned_pos_file.csv"
cleaned_neg_file = "./twitter-sentiment/data/cleaned_neg_file.csv"
cleaned_pos_test = "./twitter-sentiment/data/cleaned_pos_test.csv"
cleaned_neg_test = "./twitter-sentiment/data/cleaned_neg_test.csv"
model_dir = "./twitter-sentiment/models/"

def train_models():
    if not os.path.exists(cleaned_pos_file) or not os.path.exists(cleaned_neg_file):
        print("Cleaning raw data from {}......".format(raw_data_file))
        preprocessingConfig = data_preprocess.PreprocessingConfig()
        preprocessing = data_preprocess.DataPreprocessing(preprocessingConfig, raw_data_file, cleaned_data_dir)
        print("\nSaved cleaned data into dir: {}".format(cleaned_data_dir))
    
    print("Loading data ......")
    x_train, y_train = load_data_and_shuffle(cleaned_pos_file, cleaned_neg_file)
    x_test, y_test = load_data_and_shuffle(cleaned_pos_test, cleaned_neg_test)
    print("Finish data loading. Train set: {}, Test set: {}".format(len(y_train), len(y_test)))
    # Tokenization
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(x_train)

    # Turn x into sequence form and transform them into 2D Numpy array
    seq_x_train = tokenizer.texts_to_sequences(x_train)
    seq_x_test = tokenizer.texts_to_sequences(x_test)

    seq_x_train = sequence.pad_sequences(seq_x_train, maxlen=120)
    seq_x_test = sequence.pad_sequences(seq_x_test, maxlen=120)
    # Get input dimension
    vacab_size = len(tokenizer.word_index) + 1
    input_length = seq_x_train.shape[1]
    print("Finish Tokenization!")

    earlyStopping = EarlyStopping(monitor="val_loss", patience=2)

    def f1_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        # Calculate f1_score
        f1_score = 2.0 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_score

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    class Metrics(Callback):
        def on_train_begin(self, logs={}):
            self.val_precisions = []
            self.val_recalls = []
            self.f1s = []

        def on_epoch_end(self, epoch, logs={}):
            val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
            val_true = self.validation_data[1]
            _val_precision, _val_recall, _val_f1, support = precision_recall_fscore_support(val_true, val_predict, average='binary')
            self.val_precisions.append(_val_precision)
            self.val_recalls.append(_val_recall)
            self.f1s.append(_val_f1)

            print("- val_precision: %f - _val_recall: %f - val_f1: %f" % (_val_precision, _val_recall, _val_f1))
            return

    metrics = Metrics()
    print("Start training models...")

    ###################### CNN Model 1 ########################
    print("Building cnn model 1......")
    np.random.seed(1)
    model = Sequential()
    model.add(Embedding(vacab_size, 64, input_length=input_length))
    model.add(Conv1D(padding="same", kernel_size=3, filters=64, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score])
    print(model.summary())
    #tensorboard1 = TensorBoard(log_dir='./twitter-sentiment/model1', histogram_freq=3, write_graph=True, write_grads=True)

    print("\nFinish building cnn model 1, and start fitting cnn model 1...")
    model.fit(seq_x_train, y_train, validation_split=0.1, epochs=2, batch_size=128, verbose=1, shuffle=True, callbacks= [earlyStopping, metrics])

    print("Finish training the model 1 ans start saving the model 1...")
    # serialize mode to json
    save_model(model_dir, model, "model1")
    print("Saved the model 1 at the dir: {}".format(model_dir))

    print("Evaluate model 1 on test data")
    scores = model.evaluate(seq_x_test, y_test, verbose=1)
    print("Test: loss: {}, accuracy: {}, f1: {}".format(scores[0], scores[1], scores[2]))

    print("Finishing on cnn model 1")

    ###################### CNN Model 2 ########################
    print("Building cnn model 2......")
    np.random.seed(2)
    model = Sequential()
    model.add(Embedding(vacab_size, 128, input_length=input_length))
    model.add(Conv1D(padding="same", kernel_size=3, filters=64, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score])
    print(model.summary())
    #tensorboard2 = TensorBoard(log_dir='./twitter-sentiment/model2', histogram_freq=3, write_graph=True, write_grads=True)

    print("\nFinish building cnn model 2, and start fitting cnn model 2...")
    model.fit(seq_x_train, y_train, validation_split=0.1, epochs=5, batch_size=128, verbose=1, shuffle=True, callbacks= [earlyStopping, metrics])

    print("Finish training the model 2 ans start saving the model 2...")
    # serialize mode to json
    save_model(model_dir, model, "model2")
    print("Saved the model 2 at the dir: {}".format(model_dir))

    print("Evaluate model 2 on test data")
    scores2 = model.evaluate(seq_x_test, y_test, verbose=1)
    print("Test: loss: {}, accuracy: {}, f1: {}".format(scores2[0], scores2[1], scores2[2]))

    print("Finishing on cnn model 2")

    ###################### CNN Model 3 ########################
    print("Building CNN model 3")
    np.random.seed(3)
    model = Sequential()
    model.add(Embedding(vacab_size, 64, input_length=input_length))
    model.add(Conv1D(padding="same", kernel_size=3, filters=64, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score])
    print(model.summary())
    #tensorboard3 = TensorBoard(log_dir="./twitter-sentiment/model3", histogram_freq=3, write_graph=True, write_grads=True)

    print("Finish building CNN model 3, and start fitting the model...")
    model.fit(seq_x_train, y_train, validation_split=0.1, epochs=2, batch_size=128, verbose=1, shuffle=True, callbacks=[earlyStopping, metrics])

    print("Finish training the model 3 ans start saving the model 3...")
    # serialize mode to json
    save_model(model_dir, model, "model3")
    print("Saved the model 3 at the dir: {}".format(model_dir))

    print("Evaluate model 3 on test data")
    scores3 = model.evaluate(seq_x_test, y_test, verbose=1)
    print("Test: loss: {}, accuracy: {}, f1: {}".format(scores[0], scores3[1], scores3[2]))

    print("Finishing on CNN Model 3!")

    ###################### CNN Model 4 ########################
    print("Building CNN model 4...")
    np.random.seed(4)
    model = Sequential()
    model.add(Embedding(vacab_size, 64, input_length=input_length))
    model.add(Conv1D(padding="same", kernel_size=3, filters=64, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score])
    print(model.summary())
    
    tensorboard = TensorBoard(log_dir="./twitter-sentiment/model4", histogram_freq=3, write_graph=True, write_grads=True)

    print("Finish building CNN model 4, and start fitting the model...")
    model.fit(seq_x_train, y_train, validation_split=0.1, epochs=2, batch_size=128, verbose=1, shuffle=True, callbacks=[earlyStopping, metrics, tensorboard])

    print("Finish training the model 4 ans start saving the model 4...")
    # serialize mode to json
    save_model(model_dir, model, "model4")
    print("Saved the model 4 at the dir: {}".format(model_dir))

    print("Evaluate model 4 on test data")
    scores4 = model.evaluate(seq_x_test, y_test, verbose=1)
    print("Test: loss: {}, accuracy: {}, f1: {}".format(scores[0], scores4[1], scores4[2]))

    print("Finishing on CNN Model 4!")

def save_model(model_dir, model, model_name):
    model_json = model.to_json()
    with open(model_dir + model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_dir + model_name + ".h5")
    print("Saved the {} at the dir: {}".format(model_name, model_dir))

if __name__ == "__main__":
    train_models()






