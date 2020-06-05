import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

"""Generating and Loading Data"""

#   Batch size when loading samples to train on.
batch_size = 1000

#   Creating normal distribution training set using Numpy.
trainset = np.random.normal(2, 1, 1000000)

plt.hist(trainset, 100, density=True)
plt.show()

#   Converting training set to a Tensor.
trainset = torch.from_numpy(trainset)
print(trainset)

#   Loader that can iterate through and load data one mini batch at a time.
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
print(len(trainloader))

"""Visualisation"""

#   Define an iterator.
data_iter = iter(trainloader)

#   Getting next sample batch of the data.
values = data_iter.next()
print(values)

#   Plotting mini-batch histogram.
plt.hist(values, 10, density=True)
plt.show()

"""Models"""


#   Building the generator model. 3 layer neural network.
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x


#   Building the discriminator model.
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))


#   Function to create random noise. 1D vector of gaussian sampled random values.
def noise(size):
    n = torch.randn(size, 100)
    return n


"""Training the Models"""
#   Neural network parameters for both models.
g_input_size = 1    # Random noise coming into the generator, per output vector.
g_hidden_size = 5   # Generator complexity.
g_output_size = 1   # Size of outputted vector.
d_input_size = batch_size   # Random real data samples coming into the discriminator.
d_hidden_size = 10  # Complexity of the discriminator
d_output_size = 1   # Single dimension for real vs fake classification.

#   Activation functions for models.
generator_activation_function = torch.tanh
discriminator_activation_function = torch.sigmoid

#   Building the generator and discriminator.
G = Generator(input_size=g_input_size,
              hidden_size=g_hidden_size,
              output_size=g_output_size,
              f=generator_activation_function)
D = Discriminator(input_size=d_input_size,
                  hidden_size=d_hidden_size,
                  output_size=d_output_size,
                  f=discriminator_activation_function)

#   Learning rate for models.
lr = 1e-3

#   Creating optimisers for discriminator and generator.
d_optimiser = optim.Adam(D.parameters(), lr=lr)
g_optimiser = optim.Adam(G.parameters(), lr=lr)

#   Loss function using Binary Cross Entropy Loss.
loss = nn.BCELoss()


#   Functions to create labels of 1s and 0s with shape = size. 1 = real, 0 = fake.
def ones_target(size):
    data = torch.ones(size, 1)
    return data


def zeros_target(size):
    data = torch.zeros(size, 1)
    return data


#   Training the discriminator.
def train_discriminator(optimiser, real_data, fake_data):
    size = real_data.size(0)

    #   Reset gradients.
    optimiser.zero_grad()

    #   1) Train on real data.
    prediction_real = D(real_data)
    #   Calculate error and back-propagate.
    error_real = loss(prediction_real, ones_target(size))
    error_real.backward()

    #   2) Train on fake data.
    prediction_fake = D(fake_data)
    #   Calculate error and back-propagate.
    error_fake = loss(prediction_fake, zeros_target(size))
    error_fake.backward()

    #   3) Update weights with gradients.
    optimiser.step()

    #   Return error and predictions for real and fake inputs.
    return (error_real + error_fake), prediction_real, prediction_fake


#   Training the generator.
def train_generator(optimiser, fake_data):
    size = fake_data.size(0)

    #   Reset gradients.
    optimiser.zero_grad()

    #   Take a noise sample input and use it to generate fake data.
    prediction = G(fake_data)

    #   Calculate error and back-propagate.
    error = loss(prediction, ones_target(size))
    error.backward()

    #   Update weights with gradients.
    optimiser.step()

    #   Return error.
    return error






