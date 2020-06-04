import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


#   Mean and standard deviation of the normal distribution.
mean = 4
std_dev = 1.5

#   Creating the normal distribution dataset.
dataset = np.random.normal(mean, std_dev, 1000)

#   Visualising the dataset.
plt.hist(np.random.normal(mean, std_dev, 1000), 50, density=True)
plt.show()

#   Function to generate random normal distribution training data on the go.
def generate_trainset(mu, sigma, n):
    data = np.random.normal(mu, sigma, n)
    trainset = torch.tensor(data)
    print(trainset)


#   Building the Generator. Simple, single n neuron layer.
class Generator(nn.Module):

    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_size), int(input_size))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))


#   Building the Discriminator. Takes a normal distribution input and outputs whether it is real or not. Single neuron.
class Discriminator(nn.Module):

    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_size), 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))


#   Training the model.
def train():
    #   Parameters
    n = 100     # Size of the training set and number of neurons in the single layer Generator nn.
    lr = 1e-3   # Learning rate of the optimisers/models.

    #   Initialising the Models
    G = Generator(n)
    D = Discriminator(n)

    #   Optimisers
    g_optimiser = torch.optim.Adam(G.parameters(), lr=lr)
    d_optimiser = torch.optim.Adam(D.parameters(), lr=lr)



