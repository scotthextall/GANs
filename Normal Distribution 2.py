import torch
import matplotlib.pyplot as plt
import numpy as np

#   Batch size when loading samples to train on.
batch_size = 1000

#   Creating normal distribution training set using Numpy.
trainset = np.random.normal(2, 1, 1000000)

plt.hist(trainset, 100, density=True)
plt.show()

#   Converting training set to a Tensor.
trainset = torch.from_numpy(trainset)
print(trainset)

#   Loader that can iterate through and load data one mini batch at a time.
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
print(len(trainloader))

"""Visualisation"""

#   Define an iterator
data_iter = iter(trainloader)

#   Getting next batch of the images and labels
numbers = data_iter.next()
print(numbers)

plt.hist(numbers, 10, density=True)
plt.show()
