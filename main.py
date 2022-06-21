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

from keras.optimizers.optimizer_v2.adam import Adam

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

# Setting the converted image size
IMAGE_SIZE = [224, 224]

# Configuring training
epochs = 25
batch_size = 64

# Load dataset
train_path = "Blood Cell Dataset/dataset2-master/images/TRAIN"
valid_path = "Blood Cell Dataset/dataset2-master/images/TEST"

# Find count of data
train_files = glob(train_path + '/*/*.jp*g')
valid_files = glob(valid_path + '/*/*.jp*g')

# Find labels
folders = glob(train_path + '/*')

# Checking image data randomly
check_random = np.random.choice(valid_files)
plt.imshow(load_img(check_random))
plt.title("Random sample")
# plt.show()

print(check_random)


# Building Kernel - Modified ResNet architecture for this dataset
def identity_block(input_, kernel_size, filters):  # Create Identity Block
    f1, f2, f3 = filters
    x = Conv2D(f1, (1, 1),
               kernel_initializer='he_normal'
               )(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(f2, kernel_size, padding='same',
               kernel_initializer='he_normal'
               )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(f3, (1, 1),
               kernel_initializer='he_normal'
               )(x)
    x = BatchNormalization()(x)
    x = add([x, input_])
    x = Activation('relu')(x)
    return x


# Create Convolutional Block
def conv_block(input_,
               kernel_size,
               filters,
               strides=(2, 2)):
    f1, f2, f3 = filters
    x = Conv2D(f1, (1, 1), strides=strides,
               kernel_initializer='he_normal'
               )(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(f2, kernel_size, padding='same',
               kernel_initializer='he_normal'
               )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(f3, (1, 1),
               kernel_initializer='he_normal'
               )(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(f3, (1, 1), strides=strides,
                      kernel_initializer='he_normal'
                      )(input_)
    shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


# Custom ResNet layer architecture for this dataset
i = Input(shape=IMAGE_SIZE + [3])
x = ZeroPadding2D(padding=(3, 3))(i)
x = Conv2D(64, (7, 7),
           strides=(2, 2),
           padding='valid',
           kernel_initializer='he_normal'
           )(x)
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
x = Flatten()(x)
prediction = Dense(
    len(folders),
    activation='softmax'
)(x)

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
def preprocess_input2(ppx):
    ppx /= 127.5
    ppx -= 1
    return ppx


train_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                               zoom_range=0.2, horizontal_flip=True, vertical_flip=True,
                               preprocessing_function=preprocess_input2)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input2)

# Image augmentation using testing data for validation
test_gen = val_gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, class_mode='sparse')

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
    # plt.show()
    break

# Define training data and testing data
train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size,
    class_mode='sparse'
)
valid_generator = val_gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size,
    class_mode='sparse'
)
# Print divided data count/percentage
print("Training data: ", "{:.2f}".format(9957 / (2487 + 9957) * 100), "%")
print("Testing data: ", "{:.2f}".format(2487 / (2487 + 9957) * 100), "%")

# Using Adam optimizer
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Model Fitting
checkpoint_filepath = '/temp/checkpoint'
r = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(train_files) // batch_size,
    validation_steps=len(valid_files) // batch_size,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True),
    ]
)

# Loss plot
plt.plot(r.history['loss'], label='training loss')
plt.plot(r.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# Accuracy plot
plt.plot(r.history['accuracy'], label='training accuracy')
plt.plot(r.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()


def get_confusion_matrix(data_path, N):
    # Check ordering of data for matrix
    print("Generating confusion matrix", N)
    predictions = []
    targets = []
    i = 0
    for x, y in val_gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False,
                                            batch_size=batch_size * 2):
        i += 1
        if i % 50 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break

    cm = confusion_matrix(targets, predictions)
    return cm


cm = get_confusion_matrix(train_path, len(train_files))
print(cm)

valid_cm = get_confusion_matrix(valid_path, len(valid_files))
print(valid_cm)

import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


plot_confusion_matrix(cm, labels, title='Train confusion matrix')

np.trace(cm) / cm.sum()

plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')

np.trace(valid_cm) / valid_cm.sum()