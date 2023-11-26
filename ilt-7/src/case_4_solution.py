# =====================================================================================================
# CASE 4 
#
# This solution file is adapted from:
# https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP%20Course%20-%20Week%203%20Exercise%20Answer.ipynb
# 
# Do not use lambda layers in your model.
# 
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import pandas as pd
import urllib, csv, random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

def case_4():
    data_url = 'https://storage.googleapis.com/learning-datasets/training_cleaned.csv'
    urllib.request.urlretrieve(data_url, '../data/training_cleaned.csv')

    embedding_dim = 100
    max_length = 16
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size=16000
    test_portion=.1

    corpus = []

    num_sentences = 0

    with open("../data/training_cleaned.csv", encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
        # Your Code here. Create list items where the first item is the text, found in row[5], and the second is the label. Note that the label is a '0' or a '4' in the text. When it's the former, make
        # your label to be 0, otherwise 1. Keep a count of the number of sentences in num_sentences
            list_item=[]
            list_item.append(row[5])
            this_label=row[0]
            if this_label=='0':
                list_item.append(0)
            else:
                list_item.append(1)
            num_sentences = num_sentences + 1
            corpus.append(list_item)    
    
    sentences=[]
    labels=[]
    # random.shuffle(corpus)

    for x in range(training_size):
        sentences.append(corpus[x][0])# YOUR CODE HERE)
        labels.append(corpus[x][1])# YOUR CODE HERE)

    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)# YOUR CODE HERE)

    word_index = tokenizer.word_index # YOUR CODE HERE)
    vocab_size=len(word_index) # YOUR CODE HERE)

    sequences = tokenizer.texts_to_sequences(sentences)# YOUR CODE HERE)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)# YOUR CODE HERE)

    split = int(test_portion * training_size) # YOUR CODE HERE) 

    test_sequences = padded[0:split] # YOUR CODE HERE]
    training_sequences = padded[split:training_size] # YOUR CODE HERE]
    test_labels = labels[0:split] # YOUR CODE HERE]
    training_labels = labels[split:training_size] # YOUR CODE HERE]

    training_padded = np.array(training_sequences) # YOUR CODE HERE 
    training_labels = np.array(training_labels) # YOUR CODE HERE 
    testing_padded = np.array(test_sequences) # YOUR CODE HERE 
    testing_labels = np.array(test_labels) # YOUR CODE HERE 

    model = tf.keras.Sequential([
        # YOUR CODE HERE. 
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length), # YOUR CODE HERE 
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy']) # YOUR CODE HERE

    model.fit(training_padded, training_labels, epochs=5, validation_data=(testing_padded, testing_labels), verbose=1) # YOUR CODE HERE
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    t0 = datetime.now()
    model = case_4()
    model.save("case_4.h5")
    print(f"Total running time: {datetime.now()-t0}")
