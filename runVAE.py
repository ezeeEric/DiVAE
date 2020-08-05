# -*- coding: utf-8 -*-
"""
Run VAE

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
import torch

import numpy as np
import matplotlib.pyplot as plt

from data.loadMNIST import loadMNIST
from modelTuner import train,test
from VAE import VAE

from copy import copy
import logging
logger = logging.getLogger(__name__)

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
LATENT_DIMS = 32

torch.manual_seed(1)

train_loader,test_loader=loadMNIST(BATCH_SIZE)

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, EPOCHS+1):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)

batch_mu = np.zeros((BATCH_SIZE, LATENT_DIMS))
batch_logvar = np.zeros((BATCH_SIZE, LATENT_DIMS))

with torch.no_grad():
    for batch_idx, (x_true, label) in enumerate(test_loader):
        x_recon, mu, logvar = model(x_true)
        batch_mu = mu
        batch_logvar = logvar

#trained with list-like code
n_samples=5

plt.figure(figsize=(10, 4.5))
for i in range(n_samples):

      # plot original image
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(x_true[i].reshape((28, 28)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n_samples, i + 1 + n_samples)
    decImg=x_recon[i].reshape((28, 28))
    plt.imshow(decImg)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
fig = plt.gcf()
plt.show()
fig.savefig("./output/testVAE.png")