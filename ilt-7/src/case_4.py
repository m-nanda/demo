# =====================================================================================================
# CASE 4 
#
# This file is adapted from:
# https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP%20Course%20-%20Week%203%20Exercise%20Question.ipynb
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
    oov_tok = ""
    training_size=#Your dataset size here. Experiment using smaller values (i.e. 16000), but don't forget to train on at least 160000 to see the best effects
    test_portion=.1

    corpus = []

    num_sentences = 0

    with open("../data/training_cleaned.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
        # Your Code here. Create list items where the first item is the text, found in row[5], and the second is the label. Note that the label is a '0' or a '4' in the text. When it's the former, make
        # your label to be 0, otherwise 1. Keep a count of the number of sentences in num_sentences
            list_item=[]
            # YOUR CODE HERE
            num_sentences = num_sentences + 1
            corpus.append(list_item)
    
    
    sentences=[]
    labels=[]
    random.shuffle(corpus)

    for x in range(training_size):
        sentences.append(# YOUR CODE HERE)
        labels.append(# YOUR CODE HERE)


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(# YOUR CODE HERE)

    word_index = # YOUR CODE HERE) 
    vocab_size=len(# YOUR CODE HERE)

    sequences = tokenizer.texts_to_sequences(# YOUR CODE HERE)
    padded = pad_sequences(# YOUR CODE HERE)

    split = # YOUR CODE HERE) 

    test_sequences = padded[# YOUR CODE HERE]
    training_sequences = padded[# YOUR CODE HERE]
    test_labels = labels[# YOUR CODE HERE]
    training_labels = labels[# YOUR CODE HERE]

    training_padded = # YOUR CODE HERE 
    training_labels = # YOUR CODE HERE 
    testing_padded = # YOUR CODE HERE 
    testing_labels = # YOUR CODE HERE 

    model = tf.keras.Sequential([
        # YOUR CODE HERE. 
        tf.keras.layers.Embedding(# YOUR CODE HERE vocab_size, embedding_dim, input_length=max_length),
    ])

    model.compile(# YOUR CODE HERE loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(# YOUR CODE HERE training_padded, training_labels, epochs=10, validation_data=(testing_padded, testing_labels), verbose=1)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    t0 = datetime.now()
    model = case_4()
    model.save("case_4.h5")
    print(f"Total running time: {datetime.now()-t0}")
