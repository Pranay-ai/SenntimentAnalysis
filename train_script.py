import argparse
import os
import boto3
import numpy as np
import pandas as pd
import sagemaker
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from numpy import asarray, zeros

def download_glove_embeddings(bucket_name, object_key, destination_path):
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).download_file(object_key, destination_path)

def preprocess_text(sentence):
    sentence = sentence.lower()
    sentence = re.compile(r'<[^>]+>').sub('', sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)
    return sentence

def load_data(training_dir):
    df = pd.read_csv(os.path.join(training_dir, 'a1_IMDB_Dataset.csv'))
    X = []
    sentences = list(df['review'])
    for sen in sentences:
        X.append(preprocess_text(sen))
    y = df['sentiment']
    y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)
    return X_train, X_test, y_train, y_test

def tokenize_text(X_train, X_test):
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(X_train)
    X_train = word_tokenizer.texts_to_sequences(X_train)
    X_test = word_tokenizer.texts_to_sequences(X_test)
    vocab_length = len(word_tokenizer.word_index) + 1
    maxlen = 100
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    return X_train, X_test, vocab_length

def load_glove_embeddings(embedding_file):
    embeddings_dictionary = dict()
    glove_file = open(embedding_file, encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    return embeddings_dictionary

def create_embedding_matrix(word_tokenizer, embeddings_dictionary, vocab_length):
    embedding_matrix = zeros((vocab_length, 100))
    for word, index in word_tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

def build_lstm_model(vocab_length, embedding_matrix, maxlen):
    model = Sequential()
    embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs):
    model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

def save_model(model, sm_model_dir):
    model.save(os.path.join(sm_model_dir, 'my_model.h5'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--embedding_file', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args, _ = parser.parse_known_args()
    epochs = args.epochs
    model_dir = args.model_dir
    sm_model_dir = args.sm_model_dir
    training_dir = args.train
    embedding_file = args.embedding_file

    download_glove_embeddings(BUCKET_NAME, 'tf-imdb-data/training/a2_glove.6B.100d.txt', 'a2_glove.6B.100d.txt')
    X_train, X_test, y_train, y_test = load_data(training_dir)
    X_train, X_test, vocab_length = tokenize_text(X_train, X_test)
    embeddings_dictionary = load_glove_embeddings('a2_glove.6B.100d.txt')
    embedding_matrix = create_embedding_matrix(word_tokenizer, embeddings_dictionary, vocab_length)
    model = build_lstm_model(vocab_length, embedding_matrix, maxlen=100)
    train_model(model, X_train, y_train, X_test, y_test, epochs)
    save_model(model, sm_model_dir)
