import math
from keras.activations import sigmoid
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras import Sequential
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Input
tf.autograph.set_verbosity(0)
# from lab_utils_common import dlc
# from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)

X_train = np.array([0., 1, 2, 3, 4, 5],
                   dtype=np.float32).reshape(-1, 1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1],
                   dtype=np.float32).reshape(-1, 1)  # 2-D Matrix

set_w = np.array([[2]])
set_b = np.array([-4.5])


def sigmoidnp(w, x, b):
    g_z = np.dot(w, x) + b
    result = 1/1 + math.e**-g_z
    print(result)


print(X_train[0])
sigmoidnp(set_w, X_train[0].reshape(1, 1), set_b)
