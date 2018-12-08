import tensorflow as tf
from tensorflow import layers


def deconv2d(x, filters, kernels=(5, 5), strides=(2, 2), padding='same'):
    x = layers.conv2d_transpose(x, filters, kernels, strides, padding,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
    x = tf.nn.leaky_relu(x)
    return x


def conv2d(x, filter, kernels=(5, 5), strides=(2, 2), padding='same'):
    x = layers.conv2d(x, filter, kernels, strides, padding,
                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
    x = tf.nn.leaky_relu(x)
    return x
