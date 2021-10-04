"""
Default Engine Class for various autoencoder models.

Tested with:
- Autoencoder
"""

import torch
import os
import coffea

# Weights and Biases
import wandb
import numpy as np

from engine.engine import Engine
from utils.histHandler import HistHandler
from utils.plotting.plotCalo import plot_calo_images

from DiVAE import logging
logger = logging.getLogger(__name__)

class EngineCaloV3(Engine):

    def __init__(self, cfg=None, **kwargs):
        logger.info("Setting up engine Calo.")
        super(EngineCaloV3, self).__init__(cfg, **kwargs)
        self._hist_handler = HistHandler(cfg)

    def fit(self, epoch, is_training=True, mode="train"):
        logger.debug("Fitting model. Train mode: {0}".format(is_training))

        # Switch model between training and evaluation mode
        # Change dataloader depending on mode
        if is_training:
            self._model.train()
            data_loader = self.data_mgr.train_loader
        else:
            self._model.eval()
            if mode == "validate":
                data_loader = self.data_mgr.val_loader
            elif mode == "test":
                data_loader = self.data_mgr.test_loader
            val_loss_dict = {'epoch': epoch}

        num_batches = len(data_loader)
        log_batch_idx = max(num_batches//self._config.engine.n_batches_log_train, 1)
        num_epochs = self._config.engine.n_epochs
        num_plot_samples = self._config.engine.n_plot_samples
        total_batches = num_batches*num_epochs
        
        kl_enabled = self._config.engine.kl_enabled
        kl_annealing = self._config.engine.kl_annealing
        kl_annealing_ratio = self._config.engine.kl_annealing_ratio
        ae_enabled = self._config.engine.ae_enabled
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, (input_data, label) in enumerate(data_loader):
                self._optimiser.zero_grad()
                
                in_data_flat = [image.flatten(start_dim=1) for image in input_data]
                in_data = torch.cat(in_data_flat, dim=1)
                
                true_energy = label[0]
                
                # Scaled the raw data to GeV units
                if not self._config.data.scaled:
                    in_data = in_data/1000.
                    
                in_data = in_data.to(self._device)
                true_energy = true_energy.to(self._device).float()
                
                fwd_output=self._model((in_data, true_energy), is_training)
                pos_weight_data = torch.tensor(self._data_mgr.inv_transform(in_data.detach().cpu().numpy())/100., dtype=torch.float, device=in_data.device)
                batch_loss_dict = self._model.loss(in_data, fwd_output, pos_weight_data)
                    
                if is_training:
                    gamma = min((((epoch-1)*num_batches)+(batch_idx+1))/(total_batches*kl_annealing_ratio), 1.0)
                    if kl_enabled:
                        if kl_annealing:
                            kl_gamma = gamma
                        else:
                            kl_gamma = 1.
                    else:
                        kl_gamma = 0.
                        
                    if ae_enabled:
                        ae_gamma = 1.
                    else:
                        ae_gamma = 0.
                        
                    batch_loss_dict["gamma"] = kl_gamma
                    batch_loss_dict["epoch"] = gamma*num_epochs
                    if "hit_loss" in batch_loss_dict.keys():
                        batch_loss_dict["loss"] = ae_gamma*batch_loss_dict["ae_loss"] + kl_gamma*batch_loss_dict["kl_loss"] + batch_loss_dict["hit_loss"]
                    else:
                        batch_loss_dict["loss"] = ae_gamma*batch_loss_dict["ae_loss"] + kl_gamma*batch_loss_dict["kl_loss"]
                        
                    batch_loss_dict["loss"].backward()
                    self._optimiser.step()
                else:
                    batch_loss_dict["gamma"] = 1.0
                    batch_loss_dict["epoch"] = epoch
                    if "hit_loss" in batch_loss_dict.keys():
                        batch_loss_dict["loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["kl_loss"] + batch_loss_dict["hit_loss"]
                    else:
                        batch_loss_dict["loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["kl_loss"]
                    for key, value in batch_loss_dict.items():
                        try:
                            val_loss_dict[key] += value
                        except KeyError:
                            val_loss_dict[key] = value
                            
                    # Update the histogram
                    if self._config.data.scaled:
                        # Divide by 1000. to scale the data to GeV units
                        in_data_t = self._data_mgr.inv_transform(in_data.detach().cpu().numpy())/1000.
                        recon_data_t = self._data_mgr.inv_transform(fwd_output.output_activations.detach().cpu().numpy())/1000.
                        
                        # Samples with uniformly distributed energies - [0, 100]
                        sample_energies, sample_data = self._model.generate_samples(self._config.engine.n_valid_batch_size)
                        sample_data_t = self._data_mgr.inv_transform(sample_data.detach().cpu().numpy())/1000.
                        
                        self._hist_handler.update(in_data_t, recon_data_t, sample_data_t)
                        
                        # Samples with specific energies
                        conditioning_energies = self._config.engine.sample_energies
                        conditioned_samples = []
                        for energy in conditioning_energies:
                            sample_energies, sample_data = self._model.generate_samples(self._config.engine.n_valid_batch_size, energy)
                            sample_data_t = self._data_mgr.inv_transform(sample_data.detach().cpu().numpy())/1000.
                            conditioned_samples.append(torch.tensor(sample_data_t))
                        
                        conditioned_samples = torch.cat(conditioned_samples, dim=0).numpy()
                        self._hist_handler.update_samples(conditioned_samples)
                    else:
                        sample_energies, sample_data = self._model.generate_samples(self._config.engine.n_valid_batch_size)
                        self._hist_handler.update(in_data.detach().cpu().numpy(),
                                                  fwd_output.output_activations.detach().cpu().numpy(),
                                                  sample_data.detach().cpu().numpy())
                        
                        # Samples with specific energies
                        conditioning_energies = self._config.engine.sample_energies
                        conditioned_samples = []
                        for energy in conditioning_energies:
                            sample_energies, sample_data = self._model.generate_samples(self._config.engine.n_valid_batch_size, energy)
                            sample_data = sample_data.detach().cpu().numpy()
                            conditioned_samples.append(torch.tensor(sample_data))
                        
                        conditioned_samples = torch.cat(conditioned_samples, dim=0).numpy()
                        self._hist_handler.update_samples(conditioned_samples)
                    
                if (batch_idx % log_batch_idx) == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                                                                                          batch_idx,
                                                                                          len(data_loader),
                                                                                          100.*batch_idx/len(data_loader),
                                                                                          batch_loss_dict["loss"]))
                    
                    if (batch_idx % (num_batches//2)) == 0:
                        if self._config.data.scaled:
                            in_data = torch.tensor(self._data_mgr.inv_transform(in_data.detach().cpu().numpy()))
                            recon_data = torch.tensor(self._data_mgr.inv_transform(fwd_output.output_activations.detach().cpu().numpy()))
                            sample_energies, sample_data = self._model.generate_samples()
                            sample_data = torch.tensor(self._data_mgr.inv_transform(sample_data.detach().cpu().numpy()))
                        else:
                            # Multiply by 1000. to scale to MeV
                            in_data = in_data*1000.
                            recon_data = fwd_output.output_activations*1000.
                            sample_energies, sample_data = self._model.generate_samples()
                            sample_data = sample_data*1000.
                            
                        input_images = []
                        recon_images = []
                        sample_images = []

                        start_index = 0
                        for layer, layer_data_flat in enumerate(in_data_flat):
                            input_image = in_data[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)]
                            recon_image = recon_data[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)]
                            sample_image = sample_data[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)]
                            
                            start_index += layer_data_flat.size(1)
                            
                            input_image = input_image.reshape((-1,) + input_data[layer].size()[1:]).detach().cpu().numpy()
                            recon_image = recon_image.reshape((-1,) + input_data[layer].size()[1:]).detach().cpu().numpy()
                            sample_image = sample_image.reshape((-1,) + input_data[layer].size()[1:]).detach().cpu().numpy()
                            
                            input_images.append(input_image)
                            recon_images.append(recon_image)
                            sample_images.append(sample_image)
                        
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
            val_loss_dict = {**val_loss_dict, **self._hist_handler.get_hist_images(), **self._hist_handler.get_scatter_plots()}
            
            if self._config.save_hists:
                hist_dict = self._hist_handler.get_hist_dict()
                for key in hist_dict.keys():
                    path = os.path.join(wandb.run.dir, "{0}.coffea".format(self._config.data.particle_type + "_" + str(key)))
                    coffea.util.save(hist_dict[key], path)
                
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