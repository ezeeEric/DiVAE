# -*- coding: utf-8 -*-
"""
Unsorted helper functions

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gif
from DiVAE import logging
logger = logging.getLogger(__name__)
    
#@title Helper Functions
def plot_latent_space(zeta, label, output='', dimensions=2):
    logger.info("Plotting Latent Space")
    fig = plt.figure()
    if dimensions==0:
        i=0
        j=1
        # Create plot
        plt.title('Latent Space Representation for MNIST')
        lz0=zeta[:,i]
        lz1=zeta[:,j]
        ll=label
        df=pd.DataFrame(list(zip(lz0,lz1,ll)),columns=['z0','z1','label'])
        for l in range(10):
            maskeddf=df.loc[df.label==l]
            plt.scatter(maskeddf['z0'], maskeddf['z1'], alpha=0.5, s=10, label=l, cmap="inferno")
        plt.xlabel(r"$\zeta_{0}$ ".format(i))
        plt.ylabel(r"$\zeta_{0}$ ".format(j))
        plt.legend(loc='upper right', bbox_to_anchor=(1.135,1.))
    else:
        plt.title('{0}-dimensional Latent Space Representation for MNIST'.format(dimensions))
        plt.axis('off')
        idx=1
        for i in range(0,dimensions):
            for j in range(0,dimensions):
                # Create plot
                ax = fig.add_subplot(dimensions,dimensions, idx)
                lz0=zeta[:,i]
                lz1=zeta[:,j]
                ll=label
                df=pd.DataFrame(list(zip(lz0,lz1,ll)),columns=['z0','z1','label'])
                for l in range(10):
                    maskeddf=df.loc[df.label==l]
                    ax.scatter(maskeddf['z0'], maskeddf['z1'], alpha=0.5, s=10, label=l, cmap="inferno")
                ax.set_xlabel(r"$\zeta_{0}$".format(i))
                ax.set_ylabel(r"$\zeta_{0}$ ".format(j))
                if idx==dimensions:
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.))
                idx+=1
    fig = plt.gcf()
    fig.savefig(output+".pdf")

def plot_MNIST_output(x_true, x_recon, n_samples=5, output="./output/testVAE.png"):
    plt.figure(figsize=(10, 4.5))
    for i in range(n_samples):
        # plot original image
        ax = plt.subplot(2, n_samples, i + 1)
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
  #  plt.show()
    fig.savefig(output)

#@title Helper Functions
def plot_autoencoder_outputs(model, n, dims):
    decoded_imgs = model.decode(x_test)

    # number of example digits to show
    n = 5
    plt.figure(figsize=(10, 4.5))
    for i in range(n):
        # plot original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title('Original Images')

        # plot reconstruction 
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title('Reconstructed Images')
    plt.show()

def plot_loss(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, historydf.values.max()))
    plt.title('Loss: %.3f' % history.history['loss'][-1])
    
def plot_compare_histories(history_list, name_list, plot_accuracy=True):
    dflist = []
    min_epoch = len(history_list[0].epoch)
    losses = []
    for history in history_list:
        h = {key: val for key, val in history.history.items() if not key.startswith('val_')}
        dflist.append(pd.DataFrame(h, index=history.epoch))
        min_epoch = min(min_epoch, len(history.epoch))
        losses.append(h['loss'][-1])

    historydf = pd.concat(dflist, axis=1)

    metrics = dflist[0].columns
    idx = pd.MultiIndex.from_product([name_list, metrics], names=['model', 'metric'])
    historydf.columns = idx
    
    plt.figure(figsize=(6, 8))

    ax = plt.subplot(211)
    historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
    plt.title("Training Loss: " + ' vs '.join([str(round(x, 3)) for x in losses]))
    
    if plot_accuracy:
        ax = plt.subplot(212)
        historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
        plt.title("Accuracy")
        plt.xlabel("Epochs")
    
    plt.xlim(0, min_epoch-1)
    plt.tight_layout()


@gif.frame
def gif_output(x_true, x_recon, epoch=None, max_epochs=None, train_loss=-1,test_loss=-1, outpath="./output/testVAE.gif"):
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
        if i==0:
            ax.text(x=0,y=35,s="Epoch {0}/{1}. Train Loss {2:.2f}. Test Loss {3:.2f}.".format(epoch,max_epochs,train_loss,test_loss))


#     fig = plt.gcf()
#   #  plt.show()
#     fig.savefig(outdir)
