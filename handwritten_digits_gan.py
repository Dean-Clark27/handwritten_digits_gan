#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#ENGR010
#Group 3 Special Project
#Dean Clark
#dmc227 
#Imported torch and neural networks (nn) to train, matplotlib to graph the 
#results, and torchvision and transforms to perform image conversions
                                                                                l
import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

#Sets the seed so that the results can be replicated 
torch.manual_seed(111)
#device points to GPU if system has an Nvidia GPU, uses CPU otherwise
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#Prepares the training data to build the discriminator/generator
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)
#Creates the data loader
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
#Plots digits to train GAN off of
real_samples, mnist_labels = next(iter(train_loader))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
    
class Discriminator(nn.Module):
    '''
    Class to take in two dimensional input and give a 1 dimensional output, 
    creating the discriminator
    '''
    def __init__(self):
        '''Method to build the discriminator '''
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        '''Method to take in the training data to train the discriminator'''
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output
#Instantiates the discriminator to the device    
discriminator = Discriminator().to(device=device)

class Generator(nn.Module):
    '''
    Class to take in two random data points that will output a two
    dimensional data, creating the generator
    '''
    def __init__(self):
        '''Method to build the generator'''
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        '''Method to take in and train the generator'''
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output
#Instantiate the generator to the device 
generator = Generator().to(device=device)

#Initialize the learning rate, number of epochs, and loss function
lr = 0.0001
num_epochs = 100
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
#Training the loop to minimize loss function and update weights
for epoch in range(num_epochs):
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
         #Data to train the discriminator with
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(
            device=device
        )
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device
        )
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(
            device=device
        )
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )
        #Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels
        )
        loss_discriminator.backward()
        optimizer_discriminator.step()

        #Data that trains the Discriminator
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device
        )

        #Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        #Show loss function for both every epoch 
        if n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
#Feeding random samples from latent space into the generator         
latent_space_samples = torch.randn(batch_size, 100).to(device=device)
generated_samples = generator(latent_space_samples)
#Plot the generated samples to show what the GAN generated
generated_samples = generated_samples.cpu().detach()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])


# In[ ]:




