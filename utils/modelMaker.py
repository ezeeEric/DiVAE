import os
import torch
import numpy as np

from data.loadMNIST import loadMNIST

from DiVAE import logging
# from DiVAE import logging
logger = logging.getLogger(__name__)
from DiVAE import config

#import defined models
from models.autoencoder import AutoEncoder
from models.sparseAE import SparseAutoEncoder
from models.variationalAE import VariationalAutoEncoder
from models.hierarchicalVAE import HierarchicalVAE
from models.conditionalVAE import ConditionalVariationalAutoEncoder
from models.sequentialVAE import SequentialVariationalAutoEncoder
from models.discreteVAE import DiVAE

model_dict={
    "AE": AutoEncoder, 
    "sparseAE": SparseAutoEncoder,
    "VAE": VariationalAutoEncoder,
    "cVAE": ConditionalVariationalAutoEncoder,
    "sVAE": SequentialVariationalAutoEncoder,
    "HiVAE": HierarchicalVAE,
    "DiVAE": DiVAE
}

class ModelMaker(object):
    def __init__(self):
        self._model=None
        self._optimiser=None

        self.data_mgr=None

        self._default_activation_fct=None
    
    def init_model(self):
        for key, model_class in model_dict.items(): 
            if key.lower()==config.model_type.lower():
                logger.info("Initialising Model Type {0}".format(config.model_type))
                #TODO change init arguments
                self.model=model_class(
                            input_dimension=self.data_mgr.get_input_dimensions(),
                            config=config,
                            activation_fct=self._default_activation_fct)
                return self.model
        logger.error("Unknown Model Type. Make sure your model is registered in modelMaker.model_dict.")
        raise NotImplementedError

    @property
    def model(self):
        assert self._model is not None, "Model is not defined."
        return self._model

    @model.setter
    def model(self,model):
        self._model=model

    @property
    def default_activation_fct(self):
        return self._default_activation_fct

    @default_activation_fct.setter
    def default_activation_fct(self, act_fct):
        self._default_activation_fct=act_fct

    def save_model(self,config_string='test'):
        logger.info("Saving Model")
        f=open(os.path.join(config.output_path,"model_{0}.pt".format(config_string)),'wb')
        torch.save(self._model.state_dict(),f)
        f.close()
        return
    
    def save_rbm(self,config_string='test'):
        logger.info("Saving RBM")
        f=open(os.path.join(config.output_path,"rbm_{0}.pt".format(config_string)),'wb')
        print(self._model.prior)
        torch.save(self._model.prior,f)
        f.close()
        return

    def register_dataManager(self,data_mgr):
        assert data_mgr is not None, "Empty Data Mgr"
        self.data_mgr=data_mgr
        return

    def load_model(self,set_eval=True):
        logger.info("Loading Model")
        #attention: model must be defined already
        self._model.load_state_dict(torch.load(config.infile))
        #training of model
        if set_eval:
            self._model.eval()
        return
    
    def register_model(self,model):
        logger.debug("Register Model")
        self._model=model
        return

    def register_optimiser(self,optimiser):
        logger.debug("Register Model")
        self._optimiser=optimiser
        return

    def train(self, epoch):
        logger.info("Training Model")
        #set pytorch train mode
        self._model.train()

        total_train_loss = 0
        for batch_idx, (inputData, label) in enumerate(self.train_loader):
            #set gradients to zero before backprop. Needed in pytorch
            self._optimiser.zero_grad()

            #each of the architectures implement slightly different forward
            #calls and loss functions
            if config.model_type=='AE':
                outputData, zeta = self._model(inputData)
                train_loss = self._model.loss(inputData,outputData)

            elif config.model_type=='VAE':
                outputData, mu, logvar, zeta = self._model(inputData)
                train_loss = self._model.loss(inputData, outputData, mu, logvar)	
            
            elif config.model_type=='cVAE':
                outputData, mu, logvar, zeta = self._model(inputData,label)
                train_loss = self._model.loss(inputData, outputData, mu, logvar)	
                
            elif config.model_type=='sVAE':
                outputData, mu, logvar = self._model(inputData,label)
                train_loss = self._model.loss(inputData, outputData, mu, logvar)	

            elif config.model_type=='HiVAE':
                outputData, mu_list, logvar_list, zeta_list = self._model(inputData)
                train_loss = self._model.loss(inputData, outputData, mu_list, logvar_list)	

            elif config.model_type=='DiVAE':
                outputData, output_activations, output_distribution,\
                         posterior_distribution, posterior_samples = self._model(inputData)
                train_loss = self._model.loss(inputData, outputData, output_activations, output_distribution, posterior_distribution, posterior_samples)
            else:
                logger.debug("ERROR Unknown Model Type")
                raise NotImplementedError

            train_loss.backward()
            total_train_loss += train_loss.item()
            self._optimiser.step()
            
            # Output logging
            if batch_idx % 100 == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(inputData), len(self.train_loader.dataset),
                    100.*batch_idx/len(self.train_loader), train_loss.data.item()/len(inputData)))
        
        total_train_loss /= len(self.train_loader.dataset)
        logger.info("Train Loss: {0}".format(total_train_loss))
        return total_train_loss
    
    def test(self):
        logger.info("Testing Model")
        self._model.eval()

        test_loss = 0
        zeta_list=None
        label_list=None

        with torch.no_grad():
            for batch_idx, (inputData, label) in enumerate(self.test_loader):
                if config.model_type=='AE':
                    outputData, zeta = self._model(inputData)
                    test_loss += self._model.loss(inputData,outputData)
                    
                    #for plotting
                    zeta_list=zeta.detach().numpy() if zeta_list is None else np.append(zeta_list,zeta.detach().numpy(),axis=0) 
                    label_list=label.detach().numpy() if label_list is None else np.append(label_list,label.detach().numpy(),axis=0) 
                
                elif config.model_type=='VAE':
                    outputData, mu, logvar, zeta = self._model(inputData)
                    test_loss += self._model.loss(inputData, outputData, mu, logvar)
                    
                    #for plotting
                    zeta_list=zeta.detach().numpy() if zeta_list is None else np.append(zeta_list,zeta.detach().numpy(),axis=0) 
                    label_list=label.detach().numpy() if label_list is None else np.append(label_list,label.detach().numpy(),axis=0) 
                
                elif config.model_type=='cVAE':
                    outputData, mu, logvar, zeta = self._model(inputData,label)
                    test_loss += self._model.loss(inputData, outputData, mu, logvar)	
                
                elif config.model_type=='sVAE':
                    outputData, mu, logvar = self._model(inputData,label)
                    test_loss += self._model.loss(inputData, outputData, mu, logvar)	
                
                elif config.model_type=='HiVAE':
                    outputData, mu_list, logvar_list, zeta_hierarchy_list = self._model(inputData)
                    test_loss += self._model.loss(inputData, outputData, mu_list, logvar_list)
                    for zeta in zeta_hierarchy_list:
                        zeta_list=zeta.detach().numpy() if zeta_list is None else np.append(zeta_list,zeta.detach().numpy(),axis=0) 
                    label_list=label.detach().numpy() if label_list is None else np.append(label_list,label.detach().numpy(),axis=0) 
                
                elif config.model_type=='DiVAE':
                    outputData, output_activations, output_distribution,\
                         posterior_distribution, posterior_samples = self._model(inputData)
                    # test_loss += self._model.loss(inputData, outputData, output_activations, output_distribution, posterior_distribution, posterior_samples)
                
        test_loss /= len(self.test_loader.dataset)
        logger.info("Test Loss: {0}".format(test_loss))
        return test_loss, inputData, outputData, zeta_list, label_list


if __name__=="__main__":
    logger.info("Willkommen!")
    mm=ModelMaker()
    logger.info("Success!")


