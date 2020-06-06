import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F     # Loss Function
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms     # Datasets

#   Mini-batch size
mb_size = 64

#   Transform data to tensor format (PyTorch's expected format).
transform = transforms.Compose([transforms.ToTensor()])

#   Downloading dataset and transforming it. train=True will only download training dataset.
trainset = torchvision.datasets.MNIST(root='./NewData', download=True, train=True, transform=transform)

#   Now make a loader that can iterate through and load our data one mini batch at a time.
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=mb_size)

"""Visualisation"""

#   Define an iterator
data_iter = iter(trainloader)

#   Getting next batch of the images and labels
images, labels = data_iter.next()
test = images.view(images.size(0), -1)
print(test.size())
print(images)

Z_dim = 100
X_dim = test.size(1)


def imshow(img):
    im = torchvision.utils.make_grid(img)
    npimg = im.numpy()
    print(npimg.shape)
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])

    plt.show()


imshow(images)

h_dim = 128
lr = 1E-3


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(Z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, X_dim),
            nn.Sigmoid()
        )
        self.model.apply(init_weight)

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(X_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )
        self.model.apply(init_weight)

    def forward(self, input):
        return self.model(input)


G = Generator()
D = Discriminator()

G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)

for epoch in range(20):
    G_loss_run = 0.0
    D_loss_run = 0.0
    for i, data in enumerate(trainloader):
        X, _ = data
        mb_size = X.size(0)
        X = X.view(X.size(0), -1)

        one_labels = torch.ones(mb_size, 1)
        zero_labels = torch.zeros(mb_size, 1)

        z = torch.randn(mb_size, Z_dim)
        G_sample = G(z)
        D_fake = D(G_sample)
        D_real = D(X)
        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)
        D_real_loss = F.binary_cross_entropy(D_real, one_labels)

        D_loss = D_fake_loss + D_real_loss
        D_solver.zero_grad()
        D_loss.backward()
        D_solver.step()

        z = torch.randn(mb_size, Z_dim)
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = F.binary_cross_entropy(D_fake, one_labels)
        G_solver.zero_grad()
        G_loss.backward()
        G_solver.step()

    print('Epoch: {},   G_loss: {},     D_loss: {}'.format(epoch, G_loss_run / (i + 1), D_loss_run / (i + 1)))
    samples = G(z).detach()
    samples = samples.view(mb_size, 1, 28, 28)
    imshow(samples)


