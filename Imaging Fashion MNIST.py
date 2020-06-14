import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

#   https://github.com/soumith/ganhacks -> Used to implement many GAN improvements in this script.

#   Transform function to convert Fashion-MNIST dataset to tensor.
#   Normalises the dataset to contain values between -1 and 1 -> useful for the generator activation function later on.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#   Downloading Fashion-MNIST dataset via torchvision and transforming it to tensor format.
dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST', download=True, train=True, transform=transform)

#   Number of images (from dataset) in each mini-batch sample produced by the dataloader.
mb_size = 100

#   Create a loader that can iterate through dataset, taking mini-batch samples to train the GAN on.
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=mb_size)

"""Visualisation"""

#   Define an iterator for the dataloader.
data_iter = iter(dataloader)

#   Getting next batch of the images and labels
images, labels = data_iter.next()


def imshow(img):
    im = torchvision.utils.make_grid(img)
    npimg = im.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])

    plt.show()


imshow(images)


"""Discriminator and Generator Models"""


#   Generator neural network.
class Generator(nn.Module):
    #   Takes noise vector of input_size = 100.
    #   Outputs vector of size 784 = 'flattened' 28x28 image.
    def __init__(self):
        super(Generator, self).__init__()
        input_size = 100
        output_size = 784

    #   Three hidden layers, each followed by a Leaky-ReLU non-linearity.
        self.hidden1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

    #   Output has Tanh activation function which outputs values (-1, 1) -> same range as normalised image tensor.
    #   See https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
        self.out = nn.Sequential(
            nn.Linear(1024, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x


#   Discriminator neural network.
class Discriminator(nn.Module):
    #   Each image is 28x28 in Fashion MNIST dataset, therefore input_size = 784.
    #   output_size = 1 corresponding to 1 (real) or 0 (fake) classifier values.
    def __init__(self):
        super(Discriminator, self).__init__()
        input_size = 784
        output_size = 1

    #   Three hidden layers, each followed by a Leaky-ReLU non-linearity and a Dropout layer to prevent over-fitting.
        self.hidden1 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

    #   Output layer has sigmoid activation function, values between 0 and 1 -> suitable as a classifier.
    #   See https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
        self.out = nn.Sequential(
            nn.Linear(256, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x


"""Useful Functions"""


#   'Flattens' (reshapes) 28x28 image to a 1-D vector of size 784 (28*28).
#   'Flattened' images (784 valued vectors) are outputted by the Generator and are also used as Discriminator input.
def image_to_vector(image):
    return image.view(image.size(0), 784)


#   Converts flattened images (784 valued vectors) back to 2-D 28x28 image representations.
def vector_to_image(vector):
    return vector.view(vector.size(0), 1, 28, 28)


#   Creates 1-D vector of gaussian sampled random values with mean = 0, variance = 1.
def get_noise(size):
    noise = torch.randn(size, 100)
    return noise


#   Create tensor of ones to use in loss function when training.
def ones_target(size):
    ones = torch.ones(size, 1)
    return ones


#   Create tensor of zeros to use in loss function when training.
def zeros_target(size):
    zeros = torch.zeros(size, 1)
    return zeros


"""Training the GAN"""


def train():
    #   Parameters
    g_lr = 0.0002   # Generator learning rate.
    d_lr = 0.0002   # Discriminator learning rate.
    num_epochs = 200   # Number of epochs to train on.
    d_steps = 1     # Train Discriminator d_steps times before Generator.
    g_steps = 1     # Train Generator g_steps times before Discriminator

    #   Initialising Generator and Discriminator.
    G = Generator()
    D = Discriminator()

    #   Creating optimisers to update Generator and Discriminator.
    g_optimiser = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.9, 0.999), eps=1e-8)
    d_optimiser = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.9, 0.999), eps=1e-8)

    #   Loss function (Binary Cross Entropy)
    loss = nn.BCELoss()

    #   Lists to store errors of the generator and discriminator.
    g_errors = []
    d_errors = []

    for epoch in range(num_epochs):
        #   n_batch = index number of real_image. real_image = image taken from mini-batch of real images.
        for n_batch, (real_image, _) in enumerate(dataloader):
            #   size = number of real images contained in mini-batch.
            size = real_image.size(0)
            for d_index in range(d_steps):
                """1) Train Discriminator"""
                #   Zero the gradients on each iteration.
                D.zero_grad()

                #   i) Convert real image to 1D vector of real values ('flattened' image).
                real_data = image_to_vector(real_image)

                #   ii) Train Discriminator on real 'flattened' images.
                d_predict_real = D(real_data)
                d_error_real = loss(d_predict_real, ones_target(size))
                d_error_real.backward()

                #   iii) Generate fake 'flattened' images (= size) and detach so gradients aren't calculated for Generator.
                noise = get_noise(size)
                fake_data = G(noise).detach()

                #   iv) Train Discriminator on fake 'flattened' images (no. of fake images = size).
                d_predict_fake = D(fake_data)
                d_error_fake = loss(d_predict_fake, zeros_target(size))
                d_error_fake.backward()

                #   v) Update Discriminator with stored gradients.
                d_optimiser.step()

                #   vi) Add error to list on last loop.
                if d_index == (d_steps - 1):
                    d_errors.append(d_error_real + d_error_fake)

            for g_index in range(g_steps):
                """2) Train Generator"""
                #   Zero the gradients on each iteration.
                G.zero_grad()

                #   i) Generate fake 'flattened' images (no. of fake images = size).
                noise = get_noise(size)
                fake_data = G(noise)

                #   ii) Use Discriminator to train Generator.
                dg_predict_fake = D(fake_data)
                g_error = loss(dg_predict_fake, ones_target(size))
                g_error.backward()

                #   iii) Update Generator with stored gradients.
                g_optimiser.step()

                #   iv) Add error to list on last loop.
                if g_index == (g_steps - 1):
                    g_errors.append(g_error)

        """4) Visualisation"""
        test_images = vector_to_image(G(get_noise(mb_size)).detach())
        test_images = test_images.view(mb_size, 1, 28, 28)
        imshow(test_images)
        plt.plot(g_errors, label='Generator Error')
        plt.plot(d_errors, label='Discriminator Error')
        plt.legend(loc='upper left')
        plt.show()


train()