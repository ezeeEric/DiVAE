"""
Default Engine Class for various autoencoder models.

Tested with:
- Autoencoder
"""

import torch

# Weights and Biases
import wandb
import numpy as np

from engine.engine import Engine
from utils.hists.histHandler import HistHandler
from utils.plotting.plotCalo import plot_calo_images

from DiVAE import logging
logger = logging.getLogger(__name__)

class EngineCalo(Engine):

    def __init__(self, cfg=None, **kwargs):
        logger.info("Setting up engine Calo.")
        super(EngineCalo, self).__init__(cfg, **kwargs)
        
        self._hist_handler = HistHandler(cfg)

    def fit(self, epoch, is_training=True):
        logger.debug("Fitting model. Train mode: {0}".format(is_training))

        # Switch model between training and evaluation mode
        # Change dataloader depending on mode
        if is_training:
            self._model.train()
            data_loader = self.data_mgr.train_loader
        else:
            self._model.eval()            
            data_loader = self.data_mgr.val_loader
            val_loss_dict = {'epoch': epoch}

        num_batches = len(data_loader)
        log_batch_idx = max(num_batches//self._config.engine.n_batches_log_train, 1)
        num_epochs = self._config.engine.n_epochs
        num_plot_samples = self._config.engine.n_plot_samples
        total_batches = num_batches*num_epochs
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, (input_data, label) in enumerate(data_loader):
                self._optimiser.zero_grad()
                
                in_data_flat = [image.flatten(start_dim=1) for image in input_data]
                in_data = torch.cat(in_data_flat, dim=1)
                in_data = in_data.to(self._device)
                    
                fwd_output=self._model(in_data)
                batch_loss_dict = self._model.loss(in_data, fwd_output)
                    
                if is_training:
                    gamma = (((epoch-1)*num_batches)+(batch_idx+1))/total_batches
                    batch_loss_dict["gamma"] = gamma
                    batch_loss_dict["epoch"] = gamma*num_epochs
                    batch_loss_dict["loss"] = batch_loss_dict["ae_loss"] + gamma*batch_loss_dict["kl_loss"]
                    batch_loss_dict["loss"].backward()
                    self._optimiser.step()
                else:
                    batch_loss_dict["gamma"] = 1.0
                    batch_loss_dict["epoch"] = epoch
                    batch_loss_dict["loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["kl_loss"]
                    for key, value in batch_loss_dict.items():
                        try:
                            val_loss_dict[key] += value
                        except KeyError:
                            val_loss_dict[key] = value
                            
                    # Update the histogram
                    self._hist_handler.update(in_data.detach().cpu().numpy(), 
                                              fwd_output.output_activations.detach().cpu().numpy(),
                                              self._model.generate_samples(self._config.engine.n_valid_batch_size).detach().cpu().numpy())
                    
                if (batch_idx % log_batch_idx) == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                                                                                          batch_idx,
                                                                                          len(data_loader),
                                                                                          100.*batch_idx/len(data_loader),
                                                                                          batch_loss_dict["loss"]))
                    
                    if (batch_idx % (num_batches//2)) == 0:
                        samples = self._model.generate_samples()
                            
                        input_images = []
                        recon_images = []
                        sample_images = []

                        start_index = 0
                        for layer, layer_data_flat in enumerate(in_data_flat):
                            recon_image = fwd_output.output_activations[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)]
                            sample_image = samples[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)]
                            start_index += layer_data_flat.size(1)
                            
                            input_image = input_data[layer][:num_plot_samples].unsqueeze(1).squeeze(1).detach().cpu().numpy()
                            recon_image = recon_image.reshape((-1, 1) + input_data[layer].size()[1:]).squeeze(1).detach().cpu().numpy()
                            sample_image = sample_image.reshape((-1, 1) + input_data[layer].size()[1:]).squeeze(1).detach().cpu().numpy()
                             
                            input_images.append(input_image*1000.)
                            recon_images.append(recon_image*1000.)
                            sample_images.append(sample_image*1000.)
                        
                        batch_loss_dict["input"] = plot_calo_images(input_images)
                        batch_loss_dict["recon"] = plot_calo_images(recon_images)
                        batch_loss_dict["sample"] = plot_calo_images(sample_images)
                        
                        if not is_training:
                            for key in batch_loss_dict.keys():
                                if key not in val_loss_dict.keys():
                                    val_loss_dict[key] = batch_loss_dict[key]
                        
                    if is_training:
                        wandb.log(batch_loss_dict)
                        
        if not is_training:
            val_loss_dict = {**val_loss_dict, **self._hist_handler.get_hist_images()}
            self._hist_handler.clear()
                
            # Average the validation loss values over the validation set
            # Modify the logging keys to prefix with 'val_'
            for key in list(val_loss_dict.keys()):
                try:
                    val_loss_dict['val_' + str(key)] = val_loss_dict[key]/num_batches
                    val_loss_dict.pop(key)
                except TypeError:
                    val_loss_dict['val_' + str(key)] = val_loss_dict[key]
                    val_loss_dict.pop(key)
                    
            wandb.log(val_loss_dict)

if __name__=="__main__":
    logger.info("Willkommen!")
    engine=Engine()
    logger.info("Success!")