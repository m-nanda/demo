# ============================================================================================
# CASE 5
#
# It is adapted from: 
# https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Exercise%20Question.ipynb
# Build and train a neural network model using the Daily Min Temperature.csv dataset.
# Use MAE as the metrics of your neural network model.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://github.com/jbrownlee/Datasets
#
# Desired MAE < 2.
# ============================================================================================

import tensorflow as tf
import numpy as np
import csv
import urllib
from datetime import datetime

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # YOUR CODE HERE
    pass


def case_5():
    data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
    urllib.request.urlretrieve(data_url, 'daily-min-temperatures.csv')

    time_step = []
    temps = []

    with open('daily-min-temperatures.csv') as csvfile:
        # YOUR CODE HERE. READ TEMPERATURES INTO TEMPS

        series =  # YOUR CODE HERE
        time = # YOUR CODE HERE

        # DO NOT CHANGE THIS CODE
        split_time = 2500

        time_train =  # YOUR CODE HERE
        x_train =  # YOUR CODE HERE
        time_valid =  # YOUR CODE HERE
        x_valid =  # YOUR CODE HERE

        # DO NOT CHANGE THIS CODE
        window_size = 30
        batch_size = 32
        shuffle_buffer_size = 1000

        train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

        model = tf.keras.models.Sequential([
            # YOUR CODE HERE.
        ])

        optimizer = tf.keras.optimizers.SGD(lr=# YOUR CODE HERE, momentum=0.9)
        model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=optimizer,
                      metrics=["mae"])
        history = model.fit(train_set,epochs=# YOUR CODE HERE)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    t0 = datetime.now()
    model = case_5()
    model.save("case_5.h5")
    print(f"Total running time: {datetime.now()-t0}")
