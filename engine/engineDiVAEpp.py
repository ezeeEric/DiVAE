"""
Engine for DiVAEPP and derived models. 

"""

import torch

# Weights and Biases
import wandb

from engine.engine import Engine

from DiVAE import logging
logger = logging.getLogger(__name__)

class EngineDiVAEpp(Engine):

    def __init__(self, cfg=None, **kwargs):
        super(Engine,self).__init__(cfg, **kwargs)

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
                    input_data = input_data.to(self._device)
                    fwd_output=self._model(input_data)

                    # Compute model-dependent loss
                    batch_loss_dict = self._model.loss(input_data,fwd_output)

                    if is_training:
                        if self._config.model.model_type == "DiVAEPP":
                            """
                            Cheap hack to allow KL annealing in DVAE++
                            """
                            gamma = (((epoch-1)*num_batches)+(batch_idx+1))/(num_epochs*num_batches)
                            #gamma = 1.0
                            batch_loss_dict["gamma"] = gamma
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
                        recon = recon.reshape((-1,) + input_data.size()[2:]).detach().cpu().numpy()
                        batch_loss_dict["recon_img"] = [wandb.Image(img, caption="Reconstruction") for img in recon]
                        
                        samples = self._model.generate_samples()
                        samples = samples.reshape((-1,) + input_data.size()[2:]).detach().cpu().numpy()
                        batch_loss_dict["sample_img"] = [wandb.Image(img, caption="Samples") for img in samples]
                        
                        input_imgs = input_data.squeeze(1).detach().cpu().numpy()
                        batch_loss_dict["input_img"] = [wandb.Image(img, caption="Input") for img in input_imgs]
                    else:
                        batch_loss_dict.pop('recon_img', None)
                        batch_loss_dict.pop('sample_img', None)
                        batch_loss_dict.pop('input_img', None)
                    
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(
                                            epoch,
                                            batch_idx*len(input_data), 
                                            len(data_loader.dataset),
                                            100.*batch_idx/len(data_loader),
                                            batch_loss_dict["loss"]))
                    
                    wandb.log(batch_loss_dict)
        return batch_loss_dict["loss"]

