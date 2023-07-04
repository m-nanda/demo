# ============================================================================================
# CASE 5
#
# The solution is adapted from: 
# https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Exercise%20Answer.ipynb
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
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def case_5():
    data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
    urllib.request.urlretrieve(data_url, '../data/daily-min-temperatures.csv')

    time_step = []
    temps = []

    with open('../data/daily-min-temperatures.csv') as csvfile:
        # YOUR CODE HERE. READ TEMPERATURES INTO TEMPS
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        step = 0
        for row in reader:
            temps.append(float(row[1]))
            time_step.append(step)
            step=step + 1

        series =  np.array(temps) # YOUR CODE HERE
        time = np.array(time_step) # YOUR CODE HERE

        # DO NOT CHANGE THIS CODE
        split_time = 2500

        time_train =  time[:split_time]# YOUR CODE HERE
        x_train =  series[:split_time]# YOUR CODE HERE
        time_valid =  time[split_time:]# YOUR CODE HERE
        x_valid =  series[:split_time:]# YOUR CODE HERE

        # DO NOT CHANGE THIS CODE
        window_size = 30
        batch_size = 32
        shuffle_buffer_size = 1000

        train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

        model = tf.keras.models.Sequential([
            # YOUR CODE HERE.
            tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                                   strides=1, padding="causal",
                                   activation="relu",
                                   input_shape=[None, 1]),
            tf.keras.layers.LSTM(60, return_sequences=True),
            tf.keras.layers.LSTM(60, return_sequences=True),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        optimizer = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9) # YOUR CODE HERE
        model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=optimizer,
                      metrics=["mae"])
        history = model.fit(train_set,epochs=15)# YOUR CODE HERE)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    t0 = datetime.now()
    model = case_5()
    model.save("case_5.h5")
    print(f"Total running time: {datetime.now()-t0}")
