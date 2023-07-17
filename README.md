# Pokémon GANs Generator

Welcome to the Pokémon GANs Generator repository! In this project, a Generative Adversarial Network (GANs) has been trained to generate new Pokémon images using a dataset of Pokémon images.

## Overview

The goal of this project is to leverage the power of GANs to generate realistic and novel Pokémon images. The GANs architecture consists of a generator network that generates new Pokémon images and a discriminator network that distinguishes between real and generated images. By training these networks in an adversarial manner, the generator learns to produce increasingly convincing Pokémon images.

## Training the GANs

To train the GANs network and generate new Pokémon images, follow these steps:

1. **Prepare the Dataset**: Make sure the Pokémon image dataset is properly organized and accessible. Ensure that the dataset is placed in the appropriate directory.

2. **Configure the Network**: Adjust the hyperparameters and network architecture settings as per your requirements. Experiment with different settings to achieve desired results.

3. **Train the GANs**: Run the training script, which will initiate the training process. The GANs network will iteratively train the generator and discriminator networks till the specified number of epochs.

4. **Generate Pokémon Images**: Once the training is complete, you can utilize the trained generator network to generate new Pokémon images. Use the provided script and specify the number of images you want to generate.

## Architecture
**DC GAN**

## Version 2
### Dataset

The dataset in this version used for training the GANs network comprises a collection of Pokémon images **with Augmentation**.  
  
Original: 819 Pokemon images with a white background in JPG format (size 256x256).  
Augmented: 8183 Pokemon images with a white background in JPG format (size 256x256).  
Total: 9002  
  
Images containing '_ aug _' in their name are augmented images.
  
Example:  
Original image: 100.jpg  
Augmented images: 100_aug_#####.jpg  
  
[Kaggle - Pokemon Images Dataset](https://www.kaggle.com/datasets/rajatvisitme/pokemon-image-dataset-v2)

### Results

Although the results may not be as promising as expected, it is encouraging to see that the model is learning the patterns more effectively compared to the previous version (which lacked data augmentation). This indicates that progress is being made, and there is potential for even better results with some modifications to the architecture or with different models.
[Kaggle - Notebook](https://www.kaggle.com/code/rajatvisitme/pokegan-v2)
[Colab - Notebook](https://github.com/rajatvisitme/PokeGAN/blob/main/PokeGAN_v2.ipynb)

## Version 1
### Dataset

The dataset used for training the GANs network comprises a collection of Pokémon images. These images serve as the basis for the generator to learn the underlying patterns and features necessary to generate new Pokémon.  
  
[Kaggle - Pokemon Images Dataset](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset)

### Results

The results have not been satisfying.
[Kaggle - Notebook](https://www.kaggle.com/code/rajatvisitme/pokegan-1-0)
[Colab - Notebook](https://github.com/rajatvisitme/PokeGAN/blob/main/PokeGAN_1_dot_0.ipynb)

## Contributions

Contributions to this repository are welcome. If you would like to enhance the Pokémon GANs Generator or address any issues, please submit a pull request. Let's collaborate and improve this project together!
  
**Thank You**