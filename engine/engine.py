"""
Training, testing and evaluation engine.

Provides data handling, logging and wandb integration.
"""

import torch

# Weights and Biases
import wandb

from DiVAE import logging
logger = logging.getLogger(__name__)

class Engine(object):

    def __init__(self, cfg=None):
        self._config=cfg
        self._model=None
        self._optimiser=None
        self._data_mgr=None

    @property
    def model(self):
        return self._model
    
    @model.setter   
    def model(self,model):
        self._model=model

    @property
    def optimiser(self):
        return self._optimiser
    
    @optimiser.setter   
    def optimiser(self,optimiser):
        self._optimiser=optimiser
    
    @property
    def data_mgr(self):
        return self._data_mgr
    
    @data_mgr.setter   
    def data_mgr(self,data_mgr):
        self._data_mgr=data_mgr
    
    def register_dataManager(self,data_mgr):
        assert data_mgr is not None, "Empty Data Mgr"
        self.data_mgr=data_mgr
        return

    def generate_samples(self):
        #options handled in model files
        #n_samples=100
        #n_gibbs_sampling_steps
        #sampling_mode
        #nrs
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
                #forward call
                #output is a namespace with members as added in the forward call
                #and subsequently used in loss()
                with torch.autograd.set_detect_anomaly(False):
                    fwd_output=self._model(input_data)

                    # Compute model-dependent loss
                    batch_loss_dict = self._model.loss(input_data,fwd_output)

                    if is_training:
                        if self._config.model.model_type == "DiVAEPP":
                            gamma = (((epoch-1)*num_batches)+(batch_idx+1))/(num_epochs*num_batches)
                            batch_loss_dict["loss"] = batch_loss_dict["ae_loss"] + gamma*batch_loss_dict["kl_loss"]
                            batch_loss_dict["loss"].backward()
                            self._optimiser.step()
                        else:
                            batch_loss_dict["loss"].backward()
                            self._optimiser.step()

                # Output logging
                if is_training and batch_idx % 100 == 0:
                    if batch_idx % 500 == 0:
                        recon = fwd_output.output_distribution.reparameterise()
                        recon = recon.reshape((-1,) + input_data.size()[2:]).detach().numpy()
                        batch_loss_dict["recon_img"] = [wandb.Image(img, caption="Reconstruction") for img in recon]
                        samples = self._model.generate_samples()
                        samples = samples.reshape((-1,) + input_data.size()[2:]).detach().numpy()
                        batch_loss_dict["sample_img"] = [wandb.Image(img, caption="Samples") for img in samples]
                    
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
