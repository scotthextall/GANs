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


"""#   Building the Generator. Simple, single n neuron layer.
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
        return self.activation(self.dense(x))"""


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
    mu = 4.5    # Mean of the real normal distribution data.
    sigma = 1.2     # Standard deviation of the real normal distribution data.
    n = 1000     # Size of the training set and number of neurons in the single layer Generator nn.
    lr = 1e-2   # Learning rate of the optimisers/models.
    num_epochs = 100000   # Number of iterations to train models on.
    g_input_size = 1
    g_hidden_size = 50
    g_output_size = 1
    d_input_size = n
    d_hidden_size = 100
    d_output_size = 1

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
    """g_optimiser = torch.optim.Adam(G.parameters(), lr=lr)
    d_optimiser = torch.optim.Adam(D.parameters(), lr=lr)"""

    g_optimiser = torch.optim.SGD(G.parameters(), lr=lr, momentum=0.9)
    d_optimiser = torch.optim.SGD(D.parameters(), lr=lr, momentum=0.9)

    #   Loss function (Binary Cross Entropy)
    loss = nn.BCELoss()

    #   Lists to store errors of the generator and discriminator.
    g_errors = []
    d_errors = []

    #   Lists to store mu and sigma of the generated fake_data.
    mu_values = []
    sigma_values = []

    noise = get_noise(n)
    print(torch.reshape(noise, [n, 1]))
    print(noise.size())

    for epoch in range(num_epochs):
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
        d_error_real.backward()

        #   iii) Create noise and reshape, feed into Generator, output fake data.
        noise = get_noise(n)
        noise = torch.reshape(noise, [n, 1])
        fake_data = G(noise)

        #   iii) Detach previously generated fake_data so that gradients aren't calculated for Generator.
        #   Use Discriminator to predict whether fake_data_detached is real (1) or fake (0).
        #   Calculate Discriminator error (should aim to output 0 (fake) for fake_data_detached) and back-propagate.
        fake_data_detached = fake_data.detach()
        d_prediction_fake_detached = D(fake_data_detached.t())
        d_error_fake = loss(d_prediction_fake_detached, torch.zeros([1]))
        d_error_fake.backward()

        #   iv) Update d_optimiser weights with gradients.
        d_optimiser.step()

        #   v) Total error in discriminator is d_error_fake + d_error_real.
        d_error = d_error_fake + d_error_real

        """2) Train Generator"""
        #   Zero the gradients on each iteration.
        G.zero_grad()

        #   i) Feed noise into Generator, output fake data.
        fake_data = G(noise)

        #   ii) Use Discriminator to predict whether fake_data is real (1) or fake (0).
        d_prediction_fake = D(fake_data.t())

        #   iii) Calculate Generator error (should aim to get Discriminator to produce value of 1 (real) for the
        #   fake_data is generated) and back-propagate.
        g_error = loss(d_prediction_fake, torch.ones([1]))
        g_error.backward()

        #   iv) Update g_optimiser weights with gradients.
        g_optimiser.step()

        """3) Add errors and predicted mu and sigma to lists for future plotting."""
        #   Adding errors to lists.
        g_errors.append(g_error)
        d_errors.append(d_error)

        #   Calculating mu and sigma from generated fake_data (detached), then add to list.
        #   First convert fake_data to numpy array.
        fake_data_detached_np = fake_data.detach().numpy()
        mu_value = fake_data_detached_np.mean()
        sigma_value = fake_data_detached_np.std()
        mu_values.append(mu_value)
        sigma_values.append(sigma_value)

        #print(epoch)

        if epoch % 1000 == 0:
            print(epoch)

        """4) Plotting histogram of the generated fake_data to visualise how it changes as models are trained."""
        if epoch % 10000 == 0:
            plt.hist(real_data.detach().numpy(), bins=100, density=True)
            plt.hist(fake_data_detached_np, bins=100, density=True)
            plt.show()

    plt.plot(g_errors)
    plt.plot(d_errors)
    plt.show()



train()



