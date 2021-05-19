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
        super(EngineCalo ,self).__init__(cfg, **kwargs)

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
        num_plot_samples = self._config.engine.n_plot_samples

        with torch.set_grad_enabled(is_training):
            for batch_idx, (input_data, label) in enumerate(data_loader):
                #set gradients to zero before backprop. Needed in pytorch because
                #the default is to sum up gradients for successive backprop. steps.
                #that is useful for RNNs but not here.
                self._optimiser.zero_grad()
                with torch.autograd.set_detect_anomaly(False):
                    in_data_flat = [image.flatten(start_dim=1) for image in input_data]
                    in_data = torch.cat(in_data_flat, dim=1)
                    in_data = in_data.to(self._device)
                    
                    fwd_output=self._model(in_data)
                    batch_loss_dict = self._model.loss(in_data, fwd_output)
                    
                    gamma = (((epoch-1)*num_batches)+(batch_idx+1))/(num_epochs*num_batches)
                    batch_loss_dict["gamma"] = gamma
                    batch_loss_dict["loss"] = batch_loss_dict["ae_loss"] + gamma*batch_loss_dict["kl_loss"]
                    batch_loss_dict["loss"].backward()
                    
                    self._optimiser.step()
                    
                    if batch_idx == 0:
                        logger.info("engineCalo.EngineCalo.fit() : Defining upsampling layer")
                        # Define layer to upsample all images to max layer size (Easier for visualization) 
                        max_size_phi = 0
                        max_size_eta = 0
                        for layer in range(len(input_data)):
                            if input_data[layer].size(1) > max_size_phi:
                                max_size_phi = input_data[layer].size(1)
                            if input_data[layer].size(2) > max_size_eta:
                                max_size_eta = input_data[layer].size(2)
                        max_size = max(max_size_phi, max_size_eta)
                        upsample = torch.nn.Upsample(size=(max_size, max_size))
                    
                    if is_training and batch_idx % 100 == 0:
                        if batch_idx % 500 == 0:
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

                        else:
                            for layer in range(len(input_data)):
                                batch_loss_dict.pop("input_layer_{}".format(layer), None)
                                batch_loss_dict.pop("recon_layer_{}".format(layer), None)
                                batch_loss_dict.pop("sample_layer_{}".format(layer), None)
                                
                        logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(
                            epoch, batch_idx*len(input_data), len(data_loader.dataset),
                            100.*batch_idx/len(data_loader), batch_loss_dict["loss"]))
                        
                        wandb.log(batch_loss_dict)

        return batch_loss_dict["loss"]

if __name__=="__main__":
    logger.info("Willkommen!")
    engine=Engine()
    logger.info("Success!")