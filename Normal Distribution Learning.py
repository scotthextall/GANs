import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


#   Mean and standard deviation of the normal distribution.
mean = 4
std_dev = 1.5

"""#   Creating the normal distribution dataset.
dataset = np.random.normal(mean, std_dev, 1000)

#   Visualising the dataset.
plt.hist(np.random.normal(mean, std_dev, 1000), 50, density=True)
plt.show()"""


#   Function to generate random normal distribution training data on the go. Also creates '1' labels to show it's real.
def get_trainset(mu, sigma, n):
    data = np.random.normal(mu, sigma, n)
    trainset = torch.tensor(data)
    labels = [1] * n
    return labels, trainset


#   Function to create inputs to pass into generator. Generator will output fake data.
#   Linear inputs generated to make it more difficult for generator to trick discriminator.
def get_generator_input(n):
    generator_input = torch.rand(n)
    return generator_input


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
    training_steps = 1   # Number of iterations to train models on.
    batch_size = 50     # Batch size to sample data for generator.

    #   Initialising the Models
    G = Generator(n)
    D = Discriminator(n)

    #   Optimisers
    g_optimiser = torch.optim.Adam(G.parameters(), lr=lr)
    d_optimiser = torch.optim.Adam(D.parameters(), lr=lr)

    #   Loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        #   Zero the gradients on each iteration.
        g_optimiser.zero_grad()

        #   Creating random inputs for generator.
        g_input = get_generator_input(n).float()
        generated_data = G(g_input)
        print(generated_data)

        #   Generate examples of real data.
        real_labels, real_data = get_trainset(mean, std_dev, n)
        real_labels = torch.tensor(real_labels).float()
        real_data = torch.tensor(real_data).float()
        print(real_labels)
        print(real_data)

        #   Training the generator. Check understanding of this bit.
        generator_discriminator_out = D(generated_data)
        g_loss = loss(generator_discriminator_out, real_labels)
        print(g_loss)
        g_loss.backward()
        g_optimiser.step()

        #   Training the discriminator on the true vs generated data.
        d_optimiser.zero_grad()
        real_d_out = D(real_data)
        real_d_loss = loss(real_d_out, real_labels)

        #   Add .detach() to re-run loop.
        generator_discriminator_out = D(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size))
        d_loss = (real_d_loss + generator_discriminator_loss) / 2
        d_loss.backward()
        d_optimiser.step()

train()



