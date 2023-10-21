import argparse, os
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
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from numpy import asarray
from numpy import zeros




import boto3
import botocore

BUCKET_NAME = 'cpsc454sabucket' # replace with your bucket name
KEY = 'tf-imdb-data/training/a2_glove.6B.100d.txt' # replace with your object key

s3 = boto3.resource('s3')


s3.Bucket(BUCKET_NAME).download_file(KEY, 'a2_glove.6B.100d.txt')
if __name__ == '__main__':
    
    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()
    
    #Can have other hyper-params such as batch-size, which we are not defining in this case
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    #sm_model_dir: model artifacts stored here after training
    #training directory has the data for the model
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--embedding_file',type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    lr         = args.learning_rate
    model_dir  = args.model_dir
    sm_model_dir = args.sm_model_dir
    training_dir   = args.train
    embedding_file = args.embedding_file


    df = pd.read_csv(training_dir + '/a1_IMDB_Dataset.csv')
    
    ############
    #Preprocessing data
    ############
    nltk.download('stopwords')
    def preprocess_text(sentence):
        sentence = sentence.lower()
        sentence = re.compile(r'<[^>]+>').sub('',sentence)
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        sentence = pattern.sub('', sentence)
        return sentence
    X = []
    sentences = list(df['review'])
    for sen in sentences:
        X.append(preprocess_text(sen))
    y = df['sentiment']
    y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(X_train)

    X_train = word_tokenizer.texts_to_sequences(X_train)
    X_test = word_tokenizer.texts_to_sequences(X_test)
    vocab_length = len(word_tokenizer.word_index) + 1
    maxlen = 100

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    # Load GloVe word embeddings and create an Embeddings Dictionary


    embeddings_dictionary = dict()
    glove_file = open('a2_glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_file.close()

    embedding_matrix = zeros((vocab_length, 100))
    for word, index in word_tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    ###########
    #Model Training

    model = Sequential()
    embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(x=X_train, 
          y=y_train, 
          epochs=epochs,
          validation_data=(X_test, y_test), verbose=1)
    
    model.save(os.path.join(sm_model_dir, '000000001'), 'my_model.h5')


