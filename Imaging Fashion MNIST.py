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
