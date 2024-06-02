import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split
# from tensorflow import keras
from tensorflow.keras import layers, losses, Sequential
# from tensorflow.keras import datasets
# from tensorflow.keras.models import Model
# import tensorflow_probability

# create the python generator
# data 0: 4H with 300 timeslot
# data 1: 1H with 300 timeslot
# data 2: 15min with 300 timeslot

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='relu')
])

class DataPiplineController:
    def __init__(self):
        pass

    def fromGenerator(self, fn, args, output_types=(), output_shapes=()):
        df_fn = tf.data.Dataset.from_generator(fn, args, output_types=list(output_types), output_shapes=output_shapes)
        return df_fn

    def fromNumpyArr(self):
        pass


