import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


#   Mean and standard deviation of the normal distribution.
mean = 4
std_dev = 1.5


#   Function to generate random normal distribution training data on the go. Also creates '1' labels to show it's real.
def get_real_data(mu, sigma, n):
    np_array = np.random.normal(mu, sigma, n)
    dataset = torch.tensor(np_array)
    return dataset


#print(get_real_data(mean, std_dev, 1000).size())


#   Function to create noise of size n to pass into generator. Generator will output fake data of size n.
#   Linear inputs generated to make it more difficult for generator to trick discriminator.
def get_noise(n):
    noise = torch.rand(n)
    return noise


#print(get_noise(1000).size())


#   Building the Generator. Simple, single n neuron layer.
#   Takes n 'noise' values and generates n fake data values.
class Generator(nn.Module):

    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_size), int(input_size))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))


#   Building the Discriminator.
#   Takes n values and outputs whether they're real or not.
class Discriminator(nn.Module):

    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_size), 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))


#array = np.random.normal(1, 2, 1000)
#data = torch.from_numpy((array))
#print(data.size())

#noise = get_noise(1000)

#plt.hist(noise.detach().numpy(), 100, density=True)

#G = Generator(1000)
#D = Discriminator(1000)
#print(G)
#print(D)

#fake_data = G(noise)
#print(fake_data)

#plt.hist(fake_data.detach().numpy(), 100, density=True)
#plt.show()

#print(G(data.float()))


#   Training the model.
def train():
    #   Parameters
    n = 1000     # Size of the training set and number of neurons in the single layer Generator nn.
    lr = 1e-3   # Learning rate of the optimisers/models.
    num_epochs = 1   # Number of iterations to train models on.

    #   Initialising the Models
    G = Generator(n)
    D = Discriminator(n)

    #   Optimisers which update/improve models
    g_optimiser = torch.optim.Adam(G.parameters(), lr=lr)
    d_optimiser = torch.optim.Adam(D.parameters(), lr=lr)

    #   Loss function (Binary Cross Entropy)
    loss = nn.BCELoss()

    for epoch in range(num_epochs):
        #   1)  Train Generator.

        #   Zero the gradients on each iteration.
        g_optimiser.zero_grad()

        #   Create random noise, feed into Generator, output fake data.
        noise = get_noise(n)
        generated_data = G(noise)
        #print(generated_data)
        #plt.hist(generated_data.detach().numpy(), 100, density=True)
        #plt.show()


train()



