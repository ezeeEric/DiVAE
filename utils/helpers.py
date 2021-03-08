

"""
Unsorted helper functions

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
import torch
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import gif
from DiVAE import logging
logger = logging.getLogger(__name__)

from types import SimpleNamespace

class OutputContainer(SimpleNamespace):
    """ #this is to facilitate a common interface in the ModelCreator fit()
        #method: instead of having different lengths of output arguments for
        #each model, we return one namespace. The entries of this namespace
        #are used in the model's loss function as parameter.
        This is based on types.SimpleNamespace but adds a fallback.
    """
    def __getattr__(self, item):
        """Only gets invoked if item doesn't exist in namespace.

        Args:
            item (): Requested output item
        """
        try:
            return self.__dict__[item]
        except KeyError: 
            logger.error("You requested a attribute {0} from the output object but it does not exist.".format(item))
            logger.error("Did you add the attribute in the forward() call of your method?")
            items = (f"{k}" for k, v in self.__dict__.items())
            logger.error("Available attributes: {0}".format("{}({})".format(type(self).__name__, ", ".join(items))))
            raise 

    def clear(self):
        """Clears the current namespace. Safety feature.
        """
        for key,_ in self.__dict__.items():
            self.__dict__[key]=None
        return self
    
    def print(self):
        """Clears the current namespace. Safety feature.
        """
        out=[str(key) for key,_ in self.__dict__.items()]
        logger.info("OutputContainer keys: {0}".format(out))
        return

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

def plot_calo_jet_generated(output_data, n_samples=5, out_file="./output/testCalo.png", do_gif=False):
    for i in range(n_samples):

        plt.figure(figsize=(10, 3.5))

        images=[]
        for j in range(0,len(output_data)):
            x_out=output_data[j]
            if j==0:
                shape=(3,96)
            elif j==1:
                shape=(12,12)
            else:
                shape=(12,6)

            reco_image=x_out[i].reshape(shape)

            #TODO this is an arbitrary hack to scale the values for a nice legend...
            minVal=reco_image.min(1,keepdim=True)[0]*15
            minVal=reco_image.min(1,keepdim=True)[0]

            reco_image[reco_image<minVal]=0            
            ax1 = plt.subplot(1, len(output_data), j + 1)
            ax1.set_box_aspect(1)
            if j==0:
                ax1.set_ylabel(r'$\phi$ Cell ID')
            ax1.set_xlabel(r'$\eta$ Cell ID')

            im=plt.imshow(reco_image,aspect="auto",cmap="cool",interpolation="none",norm=LogNorm(None,None))
            images.append(im)

        fig = plt.gcf()
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.LogNorm(vmin=1e-5, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        cbar=fig.colorbar(images[0], ax=[fig.axes], orientation='vertical', fraction=.02)    
        cbar.ax.set_ylabel('Energy Fraction', rotation=270)

        for im in images:
            im.callbacksSM.connect('changed', update)
        fig.suptitle('Geant4 vs. sVAE Calorimeter shower')
        # plt.tight_layout()
        fig.savefig(output.replace(".png","_{0}.png".format(i)))

def plot_calo_image_sequence(input_data, output_data, input_dimension, layer=0, n_samples=5, out_file="./output/testCalo.png", do_gif=False):
    for i in range(n_samples):

        plt.figure(figsize=(10, 7))
        # plt.subplots_adjust(right=0.8)

        images=[]
        for j in range(0,len(input_data)):
            x=input_data[j]
            x_out=output_data[j]
            # plt.ylim([0,12])

            ax1 = plt.subplot(2, len(input_data), j + 1)
            ax1.set_box_aspect(1)
            if j==0:
                ax1.set_ylabel(r'$\phi$ Cell ID')
                # ax1.get_xaxis().set_visible(False)
            # else:
                # ax1.get_xaxis().set_visible(False)
                # ax1.get_yaxis().set_visible(False)

            im=plt.imshow(x[i],aspect="auto",cmap="cool",interpolation="none",norm=LogNorm(None,None))
            images.append(im)
            reco_image=x_out[i].reshape(x[i].shape)
            
            #TODO this is an arbitrary hack to scale the values for a nice legend...
            minVal=reco_image.min(1,keepdim=True)[0]*15
            minVal=reco_image.min(1,keepdim=True)[0]

            reco_image[reco_image<minVal]=0
        
            ax2 = plt.subplot(2, len(input_data), j + 1 + len(input_data))
            ax2.set_box_aspect(1)
            if j==0:
                ax2.set_ylabel(r'$\phi$ Cell ID')
            ax2.set_xlabel(r'$\eta$ Cell ID')
            # else:
            #     ax2.get_yaxis().set_visible(False)

            im2=plt.imshow(reco_image,aspect="auto",cmap="cool",interpolation="none",norm=LogNorm(None,None))
            # images.append(im2)

        fig = plt.gcf()
        # fig.subplots_adjust(right=0.8)
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.LogNorm(vmin=1e-5, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        cbar=fig.colorbar(images[0], ax=[fig.axes], orientation='vertical', fraction=.02)    
        cbar.ax.set_ylabel('Energy Fraction', rotation=270)

        for im in images:
            im.callbacksSM.connect('changed', update)
        fig.suptitle('Geant4 vs. sVAE Calorimeter shower')
        # plt.tight_layout()
        fig.savefig(out_file.replace(".png","_{0}.png".format(i)))

@gif.frame
def plot_calo_images(input_data, output_data, layer=0, n_samples=5, out_file="./output/testCalo.png", do_gif=False):
    plt.figure(figsize=(10, 4.5))
    axes_rec=[]
    axes_true=[]
    images=[]
    for i in range(n_samples):
        # plot original image
        ax1 = plt.subplot(2, n_samples, i + 1)
        im = plt.imshow(input_data[i],
        # aspect=float(96)/3,
        norm=LogNorm(None,None)
        )

        if i==0:    
            plt.ylabel(r'$\phi$ Cell ID')
            ax1.get_xaxis().set_visible(False)

        else:
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
        
        reco_image=output_data[i].reshape(input_data[i].shape)
        
        #TODO this is an arbitrary hack to scale the values for a nice legend...
        minVal=reco_image.min(1,keepdim=True)[0]*5
        minVal=reco_image.min(1,keepdim=True)[0]

        reco_image[reco_image<minVal]=0
        
        ax2 = plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(reco_image,
            norm=LogNorm(None,None),
        )
        if i==0:    
            plt.ylabel(r'$\phi$ Cell ID')
        else:
             ax2.get_yaxis().set_visible(False)
        plt.xlabel(r'$\eta$ Cell ID')
        axes_true.append(ax1)
        axes_rec.append(ax2)

    if not do_gif:
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(output)
        import sys
        sys.exit()

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

def plot_generative_output(input_data, n_samples=100, out_file="./output/testVAE.png"):
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
