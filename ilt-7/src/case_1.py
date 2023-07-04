# ====================================================
# CASE 1
#
# Predict the model with new values of X [-6.0, 18.0]
#
# Do not use lambda layers in your model.
#
# Desired loss (MSE) < 1e-5
# ====================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

def case_1():
    X = np.array([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0], dtype=float)
    Y = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    # YOUR CODE HERE

    print(model.predict([-6.0, 18.0]))
    return model

# The code below is to save your model as a .h5 file
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    t0 = datetime.now()
    model = case_1()
    model.save("case_1.h5")
    print(f"Total running time: {datetime.now()-t0}")