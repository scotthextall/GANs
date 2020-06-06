import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

"""Generating and Loading Data"""

#   Batch size when loading samples to train on.
#   No need for batches in this case.
batch_size = 1000

#   Creating normal distribution training set using Numpy.
dataset = np.random.normal(2, 1, 100000)

plt.hist(dataset, 100, density=True)
plt.show()

#   Converting training set to a Tensor.
dataset = torch.from_numpy(dataset)

#   Loader that can iterate through and load data one mini batch at a time.
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)

"""Visualisation"""

#   Define an iterator.
data_iter = iter(dataloader)

#   Getting next sample batch of the data.
test = data_iter.next()

#   Plotting mini-batch histogram.
plt.hist(test, 10, density=True)
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
def noise(m, n):
    random_input = torch.randn(m, n)
    return random_input


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
    ones = torch.ones(size, 1)
    return ones


def zeros_target(size):
    zeros = torch.zeros(size, 1)
    return zeros


#   Training the discriminator.
def train_discriminator(optimiser, real_data, fake_data):
    size = real_data.size(0)

    #   Reset gradients.
    optimiser.zero_grad()

    #   1) Train on real data.
    prediction_real = D(real_data.float())
    #   Calculate error and back-propagate.
    error_real = loss(prediction_real, ones_target(1))
    error_real.backward()

    #   2) Train on fake data.
    prediction_fake = D(fake_data.float().t())
    #   Calculate error and back-propagate.
    error_fake = loss(prediction_fake, zeros_target(1))
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
    #   Use discriminator to predict whether its real (1) or fake (0).
    prediction = D(fake_data.float().t())

    #   Calculate error and back-propagate.
    error = loss(prediction, ones_target(1))
    error.backward()

    #   Update weights with gradients.
    optimiser.step()

    #   Return error.
    return error


#   Total number of epochs to train on.
num_epochs = 100

#   Training loop.
for epoch in range(num_epochs):
    for real_batch in dataloader:
        #   Size of the batch of real data obtained from the dataloader.
        size = real_batch.size(0)

        #   1) Train Discriminator.
        real_data = real_batch

        #   Generate fake data using generator and noise and detach so that gradients aren't calculated for generator.
        fake_data_detached = G(noise(size, g_input_size)).detach()

        #   Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimiser, real_data, fake_data_detached)

        #   2) Train Generator.
        #   Generate fake data.
        fake_data = G(noise(size, g_input_size))

        #   Train G
        g_error = train_generator(g_optimiser, fake_data)

    if epoch % 10 == 0:
        plt.hist(fake_data.detach().numpy(), 10, density=True)
        plt.show()




"""for epoch in range(num_epochs):
    for real_batch in trainloader:
        print(real_batch)
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real data
        d_real_data = D(real_batch.float())
        d_real_prediction = D(d_real_data)
        d_real_error = loss(d_real_prediction, torch.ones([1, 1]))  # ones = true
        d_real_error.backward()  # compute/store gradients, but don't change params
        print(d_real_data)
    
        #  1B: Train D on fake
        d_gen_input = noise(batch_size, g_input_size)
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_prediction = D(d_fake_data)
        d_fake_error = loss(d_fake_prediction, torch.zeros([1, 1]))  # zeros = fake
        d_fake_error.backward()
        d_optimiser.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
        print(d_fake_data)
    
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()
    
        gen_input = noise(batch_size, g_input_size)
        g_fake_data = G(gen_input)
        dg_fake_decision = D(g_fake_data)
        g_error = loss(dg_fake_decision, torch.ones([1, 1]))  # Train G to pretend it's genuine
        print(g_fake_data)
    
        g_error.backward()
        g_optimiser.step()  # Only optimizes G's parameters"""










