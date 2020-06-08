import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#   Function to generate random normal distribution training data on the go. Also creates '1' labels to show it's real.
def get_real_data(mu, sigma, n):
    np_array = np.random.normal(mu, sigma, n)
    dataset = torch.tensor(np_array)
    return dataset


#   Function to create noise of size n to pass into generator. Generator will output fake data of size n.
#   Linear inputs generated to make it more difficult for generator to trick discriminator.
def get_noise(n):
    noise = torch.randn(n)
    return noise

#   Generator model. Neural network with 1 input layer, 2 hidden layers, 1 output layer.
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


#   Discriminator model. Neural network with 1 input layer, 2 hidden layers, 1 output layer.
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


#   Training the model.
def train():
    #   Parameters
    mu = 4.5    # Mean of the real normal distribution data.
    sigma = 1.2     # Standard deviation of the real normal distribution data.
    n = 25    # Size of the training batches.
    g_lr = 1e-3     # Learning rate of Generator optimiser.
    d_lr = 1e-3     # Learning rate of Discriminator optimiser.
    num_epochs = 20000     # Number of iterations to train models on.
    g_input_size = 1    # Single node in Generator input layer -> Single value noise input.
    g_hidden_size = 5   # Number of nodes in Generator hidden layers.
    g_output_size = 1   # Single node in Generator output layer -> Single fake value output.
    d_input_size = n    # n nodes in Discriminator input layer -> Takes n value input (Whole dataset).
    d_hidden_size = 10  # Number of nodes in Discriminator hidden layers.
    d_output_size = 1   # Single node in Discriminator output layer -> Single value 1 (real) or 0 (fake) output.
    d_steps = 20    # Update Discriminator d_steps times before Generator.
    g_steps = 20    # Update Generator g_steps times before Discriminator.

    #   Activation functions for the Generator and Discriminators.
    g_activation_function = torch.tanh
    d_activation_function = torch.sigmoid

    #   Initialising the Models
    G = Generator(input_size=g_input_size,
                  hidden_size=g_hidden_size,
                  output_size=g_output_size,
                  f=g_activation_function)
    D = Discriminator(input_size=d_input_size,
                      hidden_size=d_hidden_size,
                      output_size=d_output_size,
                      f=d_activation_function)

    #   Optimisers which update/improve models
    """g_optimiser = torch.optim.SGD(G.parameters(), lr=lr, momentum=0.9)
    d_optimiser = torch.optim.SGD(D.parameters(), lr=lr, momentum=0.9)"""

    g_optimiser = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.9, 0.999), eps=1e-8)
    d_optimiser = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.9, 0.999), eps=1e-8)
    #   Loss function (Binary Cross Entropy)
    loss = nn.BCELoss()

    #   Lists to store errors of the generator and discriminator.
    g_errors = []
    d_errors = []

    #   Lists to store mu and sigma of the generated fake_data.
    mu_values = []
    sigma_values = []

    for epoch in range(num_epochs):
        for d_index in range(d_steps):
            """1) Train Discriminator"""
            #   Zero the gradients on each iteration.
            D.zero_grad()

            #   i) Generate real data.
            real_data = get_real_data(mu, sigma, n)
            #   plt.hist(real_data, 100, density=True)

            #   ii) Use Discriminator to predict whether real_data is real (1) or fake (0).
            #   Calculate Discriminator error (should aim to output 1 (real) for real_data) and back-propagate.
            d_prediction_real = D(real_data.float())
            d_error_real = loss(d_prediction_real, torch.ones([1]))
            d_error_real.backward()     # Compute/Store gradients.

            #   iii) Create noise and reshape, feed into Generator, output fake data to train Discriminator on.
            noise = get_noise(n)
            noise = torch.reshape(noise, [n, 1])
            d_fake_data = G(noise)

            #   iii) Detach previously generated fake_data so that gradients aren't calculated for Generator.
            #   Use Discriminator to predict whether fake_data_detached is real (1) or fake (0).
            #   Calculate Discriminator error (should aim to output 0 (fake) for fake_data_detached) and back-propagate.
            d_fake_data_detached = d_fake_data.detach()
            d_prediction_fake_detached = D(d_fake_data_detached.t())
            d_error_fake = loss(d_prediction_fake_detached, torch.zeros([1]))
            d_error_fake.backward()     # Compute/Store gradients.

            #   iv) Update d_optimiser weights with stored gradients.
            d_optimiser.step()

            #   v) Total error in discriminator is d_error_fake + d_error_real, then add d_error to list.
            d_error = d_error_fake + d_error_real
            d_errors.append(d_error)

        for g_index in range(g_steps):
            """2) Train Generator"""
            #   Zero the gradients on each iteration.
            G.zero_grad()

            #   i) Feed noise into Generator, output fake data to use with Discriminator,
            #   which in turn will train Generator.
            noise = get_noise(n)
            noise = torch.reshape(noise, [n, 1])
            g_fake_data = G(noise)

            #   ii) Use Discriminator to predict whether fake_data is real (1) or fake (0).
            dg_prediction_fake = D(g_fake_data.t())

            #   iii) Calculate Generator error (should aim to get Discriminator to produce value of 1 (real) for the
            #   fake_data is generated) and back-propagate.
            g_error = loss(dg_prediction_fake, torch.ones([1]))
            g_error.backward()      # Compute/store gradients.

            #   iv) Update g_optimiser weights with stored gradients.
            g_optimiser.step()

            #   v) Adding errors to lists.
            g_errors.append(g_error)

        #   Calculating mu and sigma from generated fake_data (detached), then add to list.
        noise = get_noise(n)
        noise = torch.reshape(noise, [n, 1])
        fake_data_detached = G(noise).detach().numpy()
        mu_value = fake_data_detached.mean()
        sigma_value = fake_data_detached.std()
        mu_values.append(mu_value)
        sigma_values.append(sigma_value)

        if epoch % 100 == 0:
            print(epoch)

        """4) Plotting histogram of the generated fake_data to visualise how it changes as models are trained."""
        if epoch % 1000 == 0:
            print("Mean: {}     Std_Dev: {}".format(fake_data_detached.mean(), fake_data_detached.std()))
            """plt.hist(real_data.detach().numpy(), bins=100, density=True)
            plt.hist(fake_data_detached_np, bins=100, density=True)"""
            plt.hist(get_real_data(mu, sigma, 10000).numpy(), bins=100, density=True)
            noise = get_noise(10000)
            noise = torch.reshape(noise, [10000, 1])
            plt.hist(G(noise).detach().numpy(), bins=100, density=True)
            plt.plot(g_errors)
            plt.plot(d_errors)
            plt.show()

    plt.plot(g_errors)
    plt.plot(d_errors)
    plt.show()


train()
