import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

#   Transform function to convert Fashion-MNIST dataset to tensor.
transform = transforms.Compose([transforms.ToTensor()])

#   Downloading Fashion-MNIST dataset via torchvision and transforming it to tensor format.
dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST', download=True, train=True, transform=transform)

#   Number of images (from dataset) in each mini-batch sample produced by the dataloader.
mb_size = 64

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
