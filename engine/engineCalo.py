"""
Default Engine Class for various autoencoder models.

Tested with:
- Autoencoder
"""

import torch

# Weights and Biases
import wandb

from engine.engine import Engine

from DiVAE import logging
logger = logging.getLogger(__name__)

class EngineCalo(Engine):

    def __init__(self, cfg=None, **kwargs):
        logger.info("Setting up default engine.")
        super(EngineCalo, self).__init__(cfg, **kwargs)

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
        log_batch_idx = num_batches//self._config.engine.n_batches_log_train
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
                            
                if batch_idx == 0:
                    upsample = self.get_upsample_layer(input_data)
                    
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
                            start_index = start_index+layer_data_flat.size(1)
                                
                            input_image = upsample(input_data[layer][:num_plot_samples].unsqueeze(1)).squeeze(1).detach().cpu().numpy()
                            recon_image = upsample(recon_image.reshape((-1, 1) + input_data[layer].size()[1:])).squeeze(1).detach().cpu().numpy()
                            sample_image = upsample(sample_image.reshape((-1, 1) + input_data[layer].size()[1:])).squeeze(1).detach().cpu().numpy()
                                
                            input_images.append(input_image)
                            recon_images.append(recon_image)
                            sample_images.append(sample_image)
                                
                        for layer in range(len(input_data)):
                            batch_loss_dict["input_layer_{}".format(layer)] = [wandb.Image(img, caption="input") for img in input_images[layer]]
                            batch_loss_dict["recon_layer_{}".format(layer)] = [wandb.Image(img, caption="recon") for img in recon_images[layer]]
                            batch_loss_dict["sample_layer_{}".format(layer)] = [wandb.Image(img, caption="sample") for img in sample_images[layer]]
                        
                        if not is_training:
                            for key in batch_loss_dict.keys():
                                if key not in val_loss_dict.keys():
                                    val_loss_dict[key] = batch_loss_dict[key]
                            
                    if is_training:
                        wandb.log(batch_loss_dict)
                        
        if not is_training:
            # Average the validation loss values over the validation set
            # Modify the logging keys to prefix with 'val_'
            for key, value in list(val_loss_dict.items()):
                try:
                    val_loss_dict['val_' + str(key)] = val_loss_dict[key]/num_batches
                    val_loss_dict.pop(key)
                except TypeError:
                    val_loss_dict['val_' + str(key)] = val_loss_dict[key]
                    val_loss_dict.pop(key)
                    
            wandb.log(val_loss_dict)
    
    def get_upsample_layer(self, input_data):
        """
        - Define layer to upsample all images to max layer size (Easier for visualization)
        
        Args:
            input_data: list of data tensors for each layer with dimensions (batch_size * phi * eta)
        
        Returns:
            upsample_layer: torch.nn.Upsample layer which upsamples input images to the max size
                            using nearest neighbor interpolation
        """
        logger.info("engineCalo.EngineCalo.get_upsample_layer() : Defining upsampling layer")
 
        max_size_phi = 0
        max_size_eta = 0
        
        for layer in range(len(input_data)):
            if input_data[layer].size(1) > max_size_phi:
                max_size_phi = input_data[layer].size(1)
            if input_data[layer].size(2) > max_size_eta:
                max_size_eta = input_data[layer].size(2)
                
        max_size = max(max_size_phi, max_size_eta)
        return torch.nn.Upsample(size=(max_size, max_size))

if __name__=="__main__":
    logger.info("Willkommen!")
    engine=Engine()
    logger.info("Success!")