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

from modelTuner import trainDiVAE,testDiVAE,evaluateDiVAE
from modelTuner import train,test,evaluate
from diVAE import DiVAE
from helpers import plot_MNIST_output, gif_output

from copy import copy
import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

NUM_EVTS = 1000
BATCH_SIZE = 100 
EPOCHS = 10
LEARNING_RATE = 1e-3
LATENT_DIMS = 2
isVAE=True

torch.manual_seed(1)

train_loader,test_loader=loadMNIST(batch_size=BATCH_SIZE,num_evts_train=NUM_EVTS,num_evts_test=NUM_EVTS)

model = DiVAE(isVAE=isVAE, latent_dimensions=LATENT_DIMS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.print_model_info()

create_gif=True
gif_frames=[]

logger.debug("Start Epoch Loop")
if isVAE:
    for epoch in range(1, EPOCHS+1):
        train_loss = train(model, train_loader, optimizer, epoch)
        test_loss  = test(model, test_loader)
        if create_gif:
            x_true, x_recon = evaluate(model, test_loader, batch_size=BATCH_SIZE, latent_dimensions=LATENT_DIMS)
            gif_frames.append(gif_output(x_true, x_recon,epoch=epoch, max_epochs=EPOCHS, train_loss=train_loss,test_loss=test_loss))
else:
    for epoch in range(1, EPOCHS+1):
        train_loss = trainDiVAE(model, train_loader, optimizer, epoch)
        test_loss  = testDiVAE(model, test_loader)
        if create_gif:
            x_true, x_recon = evaluateDiVAE(model, test_loader, batch_size=BATCH_SIZE, latent_dimensions=LATENT_DIMS)
            gif_frames.append(gif_output(x_true, x_recon,epoch=epoch, max_epochs=EPOCHS, train_loss=train_loss,test_loss=test_loss))
logger.debug("Finished Epoch Loop")

configString="_".join(str(i) for i in [NUM_EVTS,BATCH_SIZE,EPOCHS,LEARNING_RATE,LATENT_DIMS])

gif.save(gif_frames,"./output/200804_VAEruns_{0}.gif".format(configString),duration=1000)

#possible to return many other parameters, x, x' just for plotting
x_true, x_recon = evaluate(model, test_loader, batch_size=BATCH_SIZE, latent_dimensions=LATENT_DIMS)

plot_MNIST_output(x_true, x_recon)
