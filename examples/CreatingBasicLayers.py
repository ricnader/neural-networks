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

# (size in 1000 square feet)
X_train = np.array([[1.0], [2.0]], dtype=np.float32)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)

set_w = np.array([[2]])
set_b = np.array([-4.5])


linear_layer = tf.keras.layers.Dense(units=1, activation='linear')
print(linear_layer.get_weights())

a1 = linear_layer(X_train[0].reshape(1, 1))
print(a1)

w, b = linear_layer.get_weights()
print(f"w = {w}, b={b}")

set_w = np.array([[200]])
set_b = np.array([100])


linear_layer.set_weights([set_w, set_b])


print(X_train[0])
a1 = linear_layer(X_train[0].reshape(1, 1))
print(a1)
alin = np.dot(set_w, X_train[0].reshape(1, 1)) + set_b
print(alin)


prediction_tf = linear_layer(X_train)
prediction_np = np.dot(X_train, set_w) + set_b
print("")
print(prediction_tf)
# print(prediction_np)


def sigmoidnp(w, x, b):
    g_z = np.dot(w, x) + b
    # result = 1/1 + math.e**-g_z
    z = np.clip(g_z, -500, 500)
    result = 1/1 + np.exp(-g_z)

    print(result)


# print(X_train[0])
# sigmoidnp(set_w, X_train[0].reshape(1, 1), set_b)
