# Import all required libraries
import os
import time

# For data visualization
import matplotlib.pyplot as plt

# tensorflow - to implement neural network
import tensorflow as tf

from train import train

from models import generator_model, discriminator_model

## DATASET
# Dataset link: 


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


## MODELS

## Create generator model
generator = generator_model()
## Get model summary
generator.summary()
## Create discriminator model
discriminator = discriminator_model()
# # Get model summary
discriminator.summary()


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


## OPTIMIZER
    
# Initializing the optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5)

## CHECKPOINTS

# Creating checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_n')
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator = generator,
                                 discriminator = discriminator)

## Other Parameters

# Hyperparameters
epochs = 1000
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Helper function to display training images
def generate_and_plot_images(model, epoch, test_input):

    predictions = model(test_input, training = False)

    fig = plt.figure(figsize = (8, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, :] * 0.5 + 0.5))
        plt.axis('off')

    plt.savefig('/<dir path to save training results>/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

## MODEL TRAINING

# Call the train function to train the models
gen_loss_epochs, disc_loss_epochs, real_score_list, fake_score_list = train(dataset, epochs = epochs)

## RESULTS

# Loss and Accuracy Plots
fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (12, 8))

ax1.plot(gen_loss_epochs, label = 'Generator loss', alpha = 0.5)
ax1.plot(disc_loss_epochs, label = 'Discriminator loss', alpha = 0.5)
ax1.legend()
ax1.set_title('Training Losses')

ax2.plot(real_score_list, label = 'Real_score', alpha = 0.5)
ax2.plot(fake_score_list, label = 'Fake_score', alpha = 0.5)
ax2.set_title('Accuracy Scores')
ax2.legend()

## SAVE MODELS

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
generator.save('/content/PokeGAN_1_dot_0_generator.h5')
discriminator.save('/content/PokeGAN_1_dot_0_discriminator.h5')

## TEST

# Recreate the exact same model, including its weights and the optimizer
trained_model = tf.keras.models.load_model('/content/PokeGAN_1_dot_0_generator.h5')

# Show the model architecture
trained_model.summary()

# Create some random noise for test input
test_input = tf.random.normal([num_examples_to_generate, noise_dim])

# Generate new Pokemons
predictions = trained_model(test_input, training = False)

fig = plt.figure(figsize = (8, 4))

for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow((predictions[i, :, :, :]))
    plt.axis('off')
plt.show()
