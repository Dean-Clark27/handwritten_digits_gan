This readme file was generated on 2023-11-29 by Dean Clark

GENERAL INFORMATION

Group 3 Special Project: Generative Adversarial Network (GAN) Creation using the PyTorch 
and Anacondas Libraries, and the Struggles of CycleGAN

Author/Principal Investigator Information
Engineering 010
Group 3 Special Project
Dean Clark
dmc227@lehigh.edu
Lehigh University

DATA & FILE OVERVIEW

first_gan.py
handwritten_digits_gan.py
Note: CUT Folder too large to be sent in the files, however, was just included for reference anyways, since I wasn't able to get it to work as intended
CUT can be found here:https://github.com/taesungp/contrastive-unpaired-translation with accompanying documentation

METHODOLOGICAL INFORMATION

For more information, the process of creating and training this data can be found at this website:
https://realpython.com/generative-adversarial-networks/#your-first-gan

DATA-SPECIFIC INFORMATION FOR: first_gan.py
This file creates and trains the discriminator and generator to reproduce the graph of the function y = sin(x)
Locally, the system will initialize the training data to plot points using the format (x, y) in a PyTorch tensor
Then will define and instantiate the discriminator which will organize the data sets into two separate groups, the real and the synthetic images
Then will define and instantiate the generator which will use the real data samples to create its own output from latent space
Then by defining the learning rate, number of epochs, and loss functions, 
We will use the training loop which will minimize the loss function for the generator and maximize it for the discriminator,
Train the data, then print the loss functions and show the data using a matplotlib plot

DATA-SPECIFIC INFORMATION FOR: handwritten_digits_gan.py
This file creates and trains the discriminator and generator to reproduce handwritten digits using the MNIST data set of 70,000 digits
First, we check to see if the device has a GPU compatible with CUDA, and if not, will use the CPU to train the data which will take up a lot more time and memory (RAM)
The program converts the values of images and puts the different values into a PyTorch tensor matrix to be trained on
Then will define and instantiate the discriminator which will organize the data sets into two separate groups, the real and the synthetic images
Then will define and instantiate the generator which will use the real data samples to create its own output from latent space
Then by defining the learning rate, number of epochs, and loss functions, 
We will use the training loop which will minimize the loss function for the generator and maximize it for the discriminator,
Train the data, then print the loss functions and show the data using a matplotlib plot of the digits


