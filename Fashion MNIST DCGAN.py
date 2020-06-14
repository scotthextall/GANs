import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils

"""Inputs and Parameters"""

#   Batch size for training
batch_size = 128

#   Spatial size of training images. All images are resized to this using a transformer
image_size = 64

#   Size of z latent vector (i.e. size of generator input)
nz = 100

#   Size of feature maps in generator
ngf = 64

#   Size of feature maps in discriminator
ndf = 64

#   Number of training epochs
num_epochs = 5

#   Learning rate for optimisers
lr = 0.0002

#   Beta1 hyperparameter for Adam optimisers
beta1 = 0.5

#   Number of GPUs available. Use 0 for CPU mode.
ngpu = 0


"""Data"""

#   Create the dataset
dataset = dset.FashionMNIST(root='./DCGANFashionMNIST', download=True, train=True,
                            transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                           ]))

#   Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#   Deciding device to run on
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

#   Plot some real training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.title('Training Images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()




