#!/usr/bin/env python
# coding: utf-8

# In[10]:


#ENGR010
#Group 3 Special Project
#Dean Clark
#dmc227 
#Imported torch and neural networks (nn) to train, math for pi, and matplotlib 
#to graph the results
import torch
from torch import nn
import math
import matplotlib.pyplot as plt

#Sets the seed so that the results can be replicated 
torch.manual_seed(111)

#Initializing the training data 
train_data_length = 1024
train_data = torch.zeros((train_data_length), 2)
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)
train_set = [(train_data[i], train_labels[i]) for i in 
             range(train_data_length)]
plt.plot(train_data[:, 0], train_data[:, 1], ".")
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
) 

class Discriminator(nn.Module):
    '''
    Class to take in two dimensional input and give a 1 dimensional output, 
    creating the discriminator
    '''
    def __init__(self):
        '''Method to build the discriminator '''
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        '''Method to take in the training data to train the discriminator'''
        output = self.model(x)
        return output
#Instantiate the Discriminator 
discriminator = Discriminator()

class Generator(nn.Module):
    '''
    Class to take in two random data points that will output a two
    dimensional data, creating the generator
    '''
    def __init__(self):
        '''Method to build the generator'''
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
    
    def forward(self, x):
        '''Method to take in and train the generator'''
        output = self.model(x)
        return output
#Instantiate the Generator
generator = Generator()

#Initialize the learning rate, number of epochs, and loss function
lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

#Training loop to minimize loss function and update weights
for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        #Data to train the discriminator with
        real_samples_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )
        
        #Training the discriminator 
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        #Data that trains the Discriminator
        latent_space_samples = torch.randn((batch_size, 2))

        #Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        #Show loss function for both every 10 epochs
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
#Feeding random samples from latent space into the generator          
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)
#Plot the generated samples to show what the GAN generated
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")

