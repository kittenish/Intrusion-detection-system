import tensorflow as tf
import numpy as np

def weight_variable(shape):
    
    #initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    initial = tf.zeros(shape)
    #initial = tf.truncated_normal(shape, stddev=0.1)
    #initial = tf.zeros(shape)
    return tf.Variable(initial)

def weight_variable_cnn(shape):
    
    #initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    #initial = tf.zeros(shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
    #initial = tf.zeros(shape)
    return tf.Variable(initial)

def bias_variable(shape):
    
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

def dense_to_one_hot(labels, n_classes=2):
    
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot