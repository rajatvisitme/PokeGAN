# Libraries for data preprocessing
import os
import time
import numpy as np
import pandas as pd
import random

# importing cv2 for image reading and image processing
import cv2
from google.colab.patches import cv2_imshow

# tensorflow - to implement neural network
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers

from models import generator_model, discriminator_model
## CREATE TRAINING DATA

# Defining the path to dataset directory
# 'DATADIR' will hold the path to downloaded dataset.
DATADIR = "/pokemon_jpg"

batch_size = 128
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Preprocessing function for image normalization and resizing
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH]) # Not required (images are already in fixed shape)
    return image

# Load and preprocess images from the directory
image_paths = [os.path.join(DATADIR, img) for img in os.listdir(DATADIR)]
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(tf.io.read_file)
dataset = dataset.map(preprocess_image)

# Create training batches
dataset = dataset.batch(batch_size)

## Create generator model
generator = generator_model()

## Get model summary
# generator.summary()

## Create discriminator model
discriminator = discriminator_model()

# # Get model summary
# discriminator.summary()

## HYPERPARAMETERS

## Loss Function
# Initializing the loss function
bce = tf.keras.losses.BinaryCrossentropy()

# Defining the discriminator loss
def discriminator_loss(real_output, fake_output):
  # `real_loss = bce(y_true, y_pred)`
  # Here, y_true is `tf.ones_like(real_output)` and y_pred is `real_output`
  # This operation `tf.ones_like(real_output)` returns a tensor of the same type and shape as `real_output` with all elements set to 1.
  # Now, we have y_true as a tensor of ones and y_pred as a tensor of predicted class (0 or 1 as binary classification i.e. real or fake)
  # Similarly, for `tf.zeros_like(fake_output)`, a tensor of zeros for fake class.
    real_loss = bce(tf.ones_like(real_output), real_output)
  # Here, we are comapring `tf.zeros_like(fake_output)` with fake_output.
  # Here, y_true should be zero as discriminator should be able to classify the `fake_output` as fake. So, the
  # ground truth (y_true) is tensor of zeros and comparing it with predicted (y_pred) `fake_output`.
  #
  # Here, the BCE loss calculation is being done for a batch of images but not for a single image.
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss

# Defining the generator loss
def generator_loss(fake_output):
  # gen_loss = bce(y_true, y_pred)
  # Here, y_true is `tf.ones_like(fake_output)` and y_pred is `fake_output`
  # This operation `tf.ones_like(fake_output)` returns a tensor of the same type and shape as `fake_output` with
  #  all elements set to 1. WHY? because we want generator to generate real like images which is class 1.
  # So, that's why we are comparing the predicted class with tensor of ones.
  # Now, we have y_true as a tensor of ones and y_pred as a tensor of predicted class (0 or 1 as binary classification i.e. real or fake)
  # And, here  `fake_output` signifies that the loss is being calculated for generated images.
    gen_loss = bce(tf.ones_like(fake_output), fake_output)

    return gen_loss
