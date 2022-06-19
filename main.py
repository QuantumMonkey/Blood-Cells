"""
PGA Capstone Project 2 - Blood Cell Image Classification

With a massive dataset of over 12000 images, we will train a Convolutional Neural Network to distinguish between the
4 types of blood cells: Eosinophil, Lymphocyte, Monocyte, Neutrophil.
"""

from keras.models import Sequential
import tensorflow as tf
# noinspection PyUnresolvedReferences
import tensorflow_datasets as tfds
tf.enable_eager_execution()
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam_legacy as adam
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from numpy import *
from PIL import Image
import theano

path_test = ""