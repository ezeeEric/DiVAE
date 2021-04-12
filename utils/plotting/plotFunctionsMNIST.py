"""
Plotting functions for visualisation. 

All plotting functions must follow this structure:

def myPlotFunction(data_container: helpers.OutputContainer, cfg: hydra-config object):
    return None
"""
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from DiVAE import logging
logger = logging.getLogger(__name__)

#TODO all the plotting functionality below will be removed (has been duplicated
#to utils.plotting.plotFunctions)
def plot_MNIST_output(input_data, output_data, n_samples=10, out_file="./output/testVAE.png"):
    plt.figure(figsize=(10, 4.5))
    for i in range(n_samples):
        # plot original image
        ax = plt.subplot(2, n_samples, i + 1)
        plt.imshow(input_data[i].reshape((28, 28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n_samples, i + 1 + n_samples)
        decImg=output_data[i].reshape((28, 28))
        plt.imshow(decImg)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig = plt.gcf()
    fig.savefig(out_file)

#@title Helper Functions
def plot_latent_space(zeta, label, out_file="", dimensions=2):
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
    fig.savefig(out_file+".pdf")

# Make images respond to changes in the norm of other images (e.g. via the
# "edit axis, curves and images parameters" GUI on Qt), but be careful not to
# recurse infinitely!
def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())

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
def gif_output(input_data, output_data, epoch=None, max_epochs=None, train_loss=-1,test_loss=-1):
    #trained with list-like code
    n_samples=5
    plt.figure(figsize=(10, 4.5))

    for i in range(n_samples):

        # plot original image
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(input_data[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n_samples, i + 1 + n_samples)
        decImg=output_data[i].reshape(input_data[i].shape)
        plt.imshow(decImg)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i==0:
            ax.text(x=0,y=35,s="Epoch {0}/{1}. Train Loss {2:.2f}. Test Loss {3:.2f}.".format(epoch,max_epochs,train_loss,test_loss))

    n_cols=5
    n_rows=int(n_samples/n_cols)
    fig,ax = plt.subplots(figsize=(n_cols,n_rows),nrows=n_rows, ncols=n_cols)
    ax_idx=0
    for i in range(n_samples):
        if ax_idx%n_cols==0:
            ax_idx+=1
        current_ax=plt.subplot(n_rows, n_cols , i+1)
        plt.imshow(input_data[i].reshape((28, 28)))
        plt.gray()
        current_ax.get_xaxis().set_visible(False)
        current_ax .get_yaxis().set_visible(False)
    fig = plt.gcf()
    # fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')