# -*- coding: utf-8 -*-
"""
Run diVAE

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
import torch
import sys

import numpy as np
import matplotlib.pyplot as plt
import gif
import pickle
from data.loadMNIST import loadMNIST

from utils.modelTuner import train,test,evaluate
from models.variationalAE import VariationalAutoEncoder
from utils.helpers import plot_MNIST_output, gif_output

from copy import copy
import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

NUM_EVTS = -1
n_batch_samples = 100 
n_epochs = 10
learning_rate = 1e-3
LATENT_DIMS = 16

torch.manual_seed(1)

train_loader,test_loader=loadMNIST(batch_size=n_batch_samples,num_evts_train=NUM_EVTS,num_evts_test=NUM_EVTS)

model = VAE(latent_dimensions=2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.print_model_info()

create_gif=True
gif_frames=[]

logger.debug("Start Epoch Loop")

for epoch in range(1, n_epochs+1):
    train_loss = train(model, train_loader, optimizer, epoch)
    test_loss  = test(model, test_loader)
    if create_gif:
        x_true, x_recon = evaluate(model, test_loader, batch_size=n_batch_samples, latent_dimensions=LATENT_DIMS)
        gif_frames.append(gif_output(x_true, x_recon,epoch=epoch, max_epochs=n_epochs, train_loss=train_loss,test_loss=test_loss))

logger.debug("Finished Epoch Loop")

configString="_".join(str(i) for i in [NUM_EVTS,n_batch_samples,n_epochs,learning_rate,LATENT_DIMS])

gif.save(gif_frames,"./output/200806_VAEruns_{0}.gif".format(configString),duration=100)

#possible to return many other parameters, x, x' just for plotting
x_true, x_recon = evaluate(model, test_loader, batch_size=n_batch_samples, latent_dimensions=LATENT_DIMS)

plot_MNIST_output(x_true, x_recon)
