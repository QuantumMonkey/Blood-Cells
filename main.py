"""
PGA Capstone Project 2 - Blood Cell Image Classification

With a massive dataset of over 12000 images, we will train a Convolutional Neural Network to distinguish between the
4 types of blood cells: Eosinophil, Lymphocyte, Monocyte, Neutrophil.
"""

import tensorflow as tf
# print(tf.__version__)

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
batch_size = 128

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


# Building Kernel - Modified ResNet architecture for this dataset
# Create Identity Block
def identity_block(input_, kernel_size, filters):
    f1, f2, f3 = filters

    # Applying filter f1 to x
    x = Conv2D(f1, (1, 1), kernel_initializer='he_normal')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Applying filter f2 to x
    x = Conv2D(f2, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Applying filter f3 to x
    x = Conv2D(f2, (1, 1), kernel_initializer='he_normal')(input_)
    x = BatchNormalization()(x)

    x = add([x, input_])
    x = Activation('relu')(x)
    return x


# Create Convolutional Block
def conv_block(input_, kernel_size, filters, strides=(2, 2)):
    f1, f2, f3 = filters

    # Applying filter f1 to x
    x = Conv2D(f1, (1, 1), strides=strides, kernel_initializer='he_normal')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Applying filter f2 to x
    x = Conv2D(f2, kernel_size, padding='same', kernel_initializer='he_normal')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Applying filter f3 to x
    x = Conv2D(f3, (1, 1), kernel_initializer='he_normal')(input_)
    x = BatchNormalization()(x)

    shortcut = Conv2D(f3, (1, 1), strides=strides, kernel_initializer='he_normal')(input_)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


# Custom ResNet layer architecture for this dataset
i = Input(shape=IMG_SIZE + [3])

x = ZeroPadding2D(padding=(3, 3))(i)
x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal')
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = ZeroPadding2D(padding=(1, 1))(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
x = identity_block(x, 3, [64, 64, 256])
x = identity_block(x, 3, [64, 64, 256])

x = conv_block(x, 3, [128, 128, 512])
x = identity_block(x, 3, [128, 128, 512])
x = identity_block(x, 3, [128, 128, 512])
x = identity_block(x, 3, [128, 128, 512])

# Fully connected layer
x = Flatten()(x)

prediction = Dense(len(folders), activation='softmax')(x)

print(len(folders))  # Every class has its own folder

# Object model
model = Model(inputs=i, outputs=prediction)

# Overall structure of the model
model.summary()

# Visualizing the model's structure
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# Image Augmentation
# Creating instance for ImageDataGenerator
def preprocess_input2(x):
    x /= 127.5
    x -= 1
    return x


train_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                               zoom_range=0.2, horizontal_flip=True, vertical_flip=True,
                               preprocessing_function=preprocess_input2)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input2)

# Image augmentation using testing data for validation
test_gen = val_gen.flow_from_directory(test_path, target_size=IMG_SIZE, class_mode='sparse')

# Collect labels for confusion matrix
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
    labels[v] = k

# View Data
for x, y in test_gen:
    print("min: ", x[0].min(), "max: ", x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break

