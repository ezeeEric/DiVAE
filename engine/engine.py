"""
Default Engine Class for various autoencoder models.

Tested with:
- Autoencoder
"""

import torch

# Weights and Biases
import wandb

from engine.engineBase import EngineBase

from DiVAE import logging
logger = logging.getLogger(__name__)

class Engine(EngineBase):

    def __init__(self, cfg=None, **kwargs):
        logger.info("Setting up default engine.")
        super(Engine,self).__init__(cfg, **kwargs)
        

    def generate_samples(self):
        #generate the samples. Each model has its own specific settings which
        #are read in from the self._config in the individual model files.
        output=self._model.generate_samples()
        #remove gradients and history, convert output to leaves so we can do
        #with it what we like (numpy operations for example).
        if isinstance(output,list):
            output=[out.detach() for out in output]
        else:
            output.detach()
        return output

    def fit(self, epoch, is_training=True):
        logger.debug("Fitting model. Train mode: {0}".format(is_training))
        
        #set train/eval mode and context. torch.no_grad in eval mode improves
        #memory consumption. nullcontext is necessary to have a neat code
        #structure like below.

        if is_training:
            self._model.train()
            data_loader=self.data_mgr.train_loader
        else:
            self._model.eval()            
            data_loader=self.data_mgr.test_loader

        epoch_loss_dict = {}
        num_batches = len(data_loader)
        num_epochs = self._config.engine.n_epochs

        with torch.set_grad_enabled(is_training):
            for batch_idx, (input_data, label) in enumerate(data_loader):
                #set gradients to zero before backprop. Needed in pytorch because
                #the default is to sum up gradients for successive backprop. steps.
                #that is useful for RNNs but not here.
                self._optimiser.zero_grad()
                with torch.autograd.set_detect_anomaly(False):
                    input_data = input_data.to(self._device)
                    #forward call
                    #output is a namespace with members as added in the forward call
                    #and subsequently used in loss()
                    fwd_output=self._model(input_data)

                    # Compute model-dependent loss
                    batch_loss_dict = self._model.loss(input_data,fwd_output)

                    batch_loss_dict["loss"].backward()
                    self._optimiser.step()
                    
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(
                                            epoch,
                                            batch_idx*len(input_data), 
                                            len(data_loader.dataset),
                                            100.*batch_idx/len(data_loader),
                                            batch_loss_dict["loss"]))
                    
                    wandb.log(batch_loss_dict)

        return batch_loss_dict["loss"]
    
    def evaluate(self):
        #similar to test call of fit() method but returning values
        self._model.eval()
        #do plots on test dataset           
        data_loader=self.data_mgr.test_loader

        with torch.no_grad():
            for _, (input_data, label) in enumerate(data_loader):
                # fwd_output=self._model(input_data)
                try:
                    fwd_output=self._model(input_data)
                except:
                    #TODO hack for conditionalVAE
                    fwd_output=self._model(input_data,label)

                fwd_output.input_data=input_data
                fwd_output.labels = label
        return fwd_output

if __name__=="__main__":
    logger.info("Willkommen!")
    engine=Engine()
    logger.info("Success!")
