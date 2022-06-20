"""
PGA Capstone Project 2 - Blood Cell Image Classification

With a massive dataset of over 12000 images, we will train a Convolutional Neural Network to distinguish between the
4 types of blood cells: Eosinophil, Lymphocyte, Monocyte, Neutrophil.
"""

import tensorflow as tf
#print(tf.__version__)

from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, \
    BatchNormalization, ZeroPadding2D, MaxPooling2D, Activation, add
from keras.models import Model
from keras.preprocessing import image
from keras.utils.image_utils import load_img
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import adam_v2 as Adam

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

# Setting the converted image size
IMG_SIZE = [200, 200]

# Configuring training
epochs = 10
batch_size = 200

# Load dataset
train_path = "Blood Cell Dataset/dataset2-master/images/TRAIN"
test_path = "Blood Cell Dataset/dataset2-master/images/TEST"

# Find count of data
train_files = glob(train_path + '/*/*.jp*g')
test_files = glob(test_path + '/*/*.jp*g')

# Find labels
folders = glob(train_path + '/*')

# Checking image data randomly
check_random = np.random.choice(train_files)
plt.imshow(load_img(check_random))
plt.show()

print(check_random)