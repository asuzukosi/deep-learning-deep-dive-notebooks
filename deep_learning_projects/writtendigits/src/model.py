import tensorflow as tf
import hyperparams
import keras


def build_model():
    inputs = keras.Input()
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense()(x)