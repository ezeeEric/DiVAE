# -*- coding: utf-8 -*-
"""
Autoencoders

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

#pyTorch: Open source ML library dev. mainly by Facebook's AI research lab
import torch
import torch.nn as nn

from distributions import SpikeAndExponentialSmoother
from copy import copy
import logging
logger = logging.getLogger(__name__)

#Base Class
class Network(nn.Module):
    def __init__(self, node_sequence=None, activation_fct=None, create_module_list=True,**kwargs):
        super(Network, self).__init__(**kwargs)
        self._layers=nn.ModuleList([]) if create_module_list else None
        self._node_sequence=node_sequence
        self._activation_fct=activation_fct

        if self._node_sequence and create_module_list:
            self._create_network()
    
    def encode(self):
        raise NotImplementedError
    
    def decode(self):
        raise NotImplementedError
    
    def _create_network(self):        
        for node in self._node_sequence:
            self._layers.append(
                nn.Linear(node[0],node[1])
            )
        return

    def get_activation_fct(self):        
        return "{0}".format(self._activation_fct).replace("()","")

#Implements encode()
class BasicEncoder(Network):
    def __init__(self,**kwargs):
        super(BasicEncoder, self).__init__(**kwargs)

    def encode(self, x):
        logger.debug("encode")
        for layer in self._layers:
            if self._activation_fct:
                x=self._activation_fct(layer(x))
            else:
                x=layer(x)
        return x

#Implements decode()
class BasicDecoder(Network):
    def __init__(self,output_activation_fct=nn.Sigmoid(),**kwargs):
        super(BasicDecoder, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct

    def decode(self, x):
        logger.debug("Decoder::decode")
        nr_layers=len(self._layers)
        for idx,layer in enumerate(self._layers):
            if idx==nr_layers-1 and self._output_activation_fct:
                x=self._output_activation_fct(layer(x))
            else:
                x=self._activation_fct(layer(x))
        return x

class SimpleEncoder(Network):
    def __init__(self,smoothing_distribution=SpikeAndExponentialSmoother(beta=4),num_latent_hierarchy_levels=4,**kwargs):
        super(SimpleEncoder, self).__init__(**kwargs)
        self.smoothing_distribution=smoothing_distribution

        #number of hierarchy levels in encoder. This is the number of latent
        #layers. At each hiearchy level an output layer is formed.
        self.num_latent_hierarchy_levels=4
        #number of latent units in the prior - output units for each level of
        #the hierarchy. Also number of input nodes to the decoder, first layer
        self.num_latent_units=100
        #each hierarchy has NN with num_det_layers_enc layers
        #number of deterministic units in each encoding layer. These layers map
        #input to the latent layer. 
        self.num_det_units=200
        # number of deterministic layers in each conditional p(z_i | z_{k<i})
        self.num_det_layers=2 

    def encode(self, x):
        logger.debug("encode")
        for layer in self._layers:
            if self._activation_fct:
                x=self._activation_fct(layer(x))
            else:
                x=layer(x)
        return x
    
    def hierarchical_posterior(self,x, is_training=True):
        """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
            to num_latent_hierarchy_levels and each element in the list is a DistUtil object containing posterior distribution 
            for the group of latent units in each hierarchy level. 

        Args:
            input: a tensor containing input tensor.
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            posterior: a list of DistUtil objects containing posterior parameters.
            post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
        """
        logger.debug("ERROR Encoder::hierarchical_posterior")
        posterior = []
        post_samples = []
        #TODO switched off hierarchy for now.
        # import pickle
        for i in range(self.num_latent_hierarchy_levels):
            # network_input = tf.concat(axis=-1, values=[input] + post_samples)  # concat x, z0, z1, ...
            # network = self.nets[i]
            # param = network.build_network(network_input, is_training)                      # create network
            # In the evaluation, we will use Bernoulli instead of continuous relaxations.
           # if not is_training and self.dist_util in {MixtureNormal, Spike_and_Exp, MixtureGeneric}:
          #      posterior_dist = FactorialBernoulliUtil([param[0]])
          #  else:
            qprime=self.encode(x)
            # pickle.dump(qprime,open( "datasample.pkl", "wb" ))
            sigmoid=torch.nn.Sigmoid()
            q=sigmoid(qprime)
            #returns tensor of size n of random variables drawn from uniform

            #dist in [0,1)
            rho=torch.rand(q.size())
            posterior_dist = self.smoothing_distribution # init posterior dist.
            samples=posterior_dist.icdf(rho,q)
            posterior.append(posterior_dist)
            post_samples.append(samples)
        return posterior, post_samples
    
class SimpleDecoder(Network):
    def __init__(self,output_nodes=None,output_activation_fct=None,**kwargs):
        super(SimpleDecoder, self).__init__(**kwargs) 
        #last output layer treated separately, as it needs sigmoid activation        

        self._output_activation_fct=output_activation_fct

    def decode(self, z):
        logger.debug("Decoder::decode")
        nr_layers=len(self._layers)
        x_prime=None
        for idx,layer in enumerate(self._layers):
            if idx==nr_layers-1:
                if self._output_activation_fct:
                    x_prime=self._output_activation_fct(layer(z))
                else:
                    x_prime=self._activation_fct(layer(z))
            else:
                z=self._activation_fct(layer(z))
        return x_prime

    def decode_posterior_sample(self, zeta):
        logger.debug("Decoder::decode")  
        nr_layers=len(self._layers)
        for idx,layer in enumerate(self._layers):
            # print(idx, layer)
            if idx==nr_layers-1:

                x_prime=self._output_activation_fct(layer(zeta))
            else:
                zeta=self._activation_fct(layer(zeta))
        return x_prime

#TODO make this inheriting from base class
class HierarchicalEncoder(nn.Module):
    def __init__(self, 
        activation_fct=nn.Tanh(),
        num_input_units=784,
        num_latent_hierarchy_levels=4,
        num_latent_units=100,
        num_det_units=200,
        num_det_layers=2,
        use_gaussian=False, 
        **kwargs):
        super(HierarchicalEncoder, self).__init__(**kwargs)
        
        #TODO
        #batch normalisation
        #weight decay

        self.smoothing_distributions=None #smoothing_distribution

        self.num_input_units=num_input_units

        #number of hierarchy levels in encoder. This is the number of latent
        #layers. At each hiearchy level an output layer is formed.
        self.num_latent_hierarchy_levels=num_latent_hierarchy_levels

        #number of latent units in the prior - output units for each level of
        #the hierarchy. Also number of input nodes to the decoder, first layer
        self.num_latent_units=num_latent_units

        #each hierarchy has NN with num_det_layers_enc layers
        #number of deterministic units in each encoding layer. These layers map
        #input to the latent layer. 
        self.num_det_units=num_det_units
        
        # number of deterministic layers in each conditional p(z_i | z_{k<i})
        self.num_det_layers=num_det_layers

        # for all layers except latent (output)
        self.activation_fct=activation_fct

        #list of all networks in the hierarchy of the encoder
        self._networks=nn.ModuleList([])
        
        #switch for HiVAE model
        self.use_gaussian=use_gaussian

        #for each hierarchy level create a network. Input units will increase
        #per level.
        for lvl in  range(self.num_latent_hierarchy_levels):
            network=self._create_hierarchy_network(level=lvl, skip_latent_layer=use_gaussian)
            self._networks.append(network)

    def _create_hierarchy_network(self,level=0, skip_latent_layer=False):       
        #TODO this should be revised with better structure for input layer config  
        layers=[self.num_input_units+level*self.num_latent_units]+[self.num_det_units]*self.num_det_layers+[self.num_latent_units]
        if skip_latent_layer:
            layers=[self.num_input_units+level*self.num_latent_units]+[self.num_det_units]*self.num_det_layers

        moduleLayers=nn.ModuleList([])
        for l in range(len(layers)-1):
            n_in_units=layers[l]
            n_out_units=layers[l+1]

            moduleLayers.append(nn.Linear(n_in_units,n_out_units))
            #apply the activation function for all layers except the last
            #(latent) layer 
            act_fct = nn.Identity() if l==len(layers)-2 else self.activation_fct
            moduleLayers.append(act_fct)

        sequential=nn.Sequential(*moduleLayers)
        return sequential

    def encode(self, x):
        logger.debug("encode")
        for layer in self._layers:
            if self._activation_fct:
                x=self._activation_fct(layer(x))
            else:
                x=layer(x)
        return x
    
    def hierarchical_posterior(self, data_input, is_training=True):
        logger.debug("ERROR Encoder::hierarchical_posterior")
        hierarchical_posterior_dist_list = []
        hierarchical_posterior_samples = []
        #Loop all levels in hierarchy
        for i in range(self.num_latent_hierarchy_levels):
            #the input x concatenated with z0,z1,z2,...
            hierarchical_input=torch.cat((data_input,*hierarchical_posterior_samples),1)
            #the network of the current hierarchy level
            current_network=self._networks[i]
            #raw output (logit) of the current NN
            encoder_logit=current_network(hierarchical_input)
            #pick the smoothing distribution to be used
            #use spike and exponential as presented by Rolfe
            posterior_distribution=SpikeAndExponentialSmoother(encoder_logit) if is_training else None
            #do the sampling step on the raw output above
            hierarchical_samples=posterior_distribution.reparameterise()
            hierarchical_posterior_samples.append(hierarchical_samples)
            hierarchical_posterior_dist_list.append(posterior_distribution)

        return hierarchical_posterior_dist_list, hierarchical_posterior_samples
    
    def get_activation_fct(self):        
        return

class Decoder(Network):
    def __init__(self,**kwargs):
        super(Decoder, self).__init__(**kwargs) 

        #TODO make this steerable: num_latent_units=num_latent_units_enc*num_latent_layers
        self.num_latent_units=400
        self.num_det_units=200
        self.num_det_layers=2
        self.num_input_units=784
        self.activation_fct=nn.Tanh()
        
        self._network=self._create_network()

    def _create_network(self):
        layers=[self.num_latent_units]+[self.num_det_units]*self.num_det_layers+[self.num_input_units]
        moduleLayers=nn.ModuleList([])
        
        for l in range(len(layers)-1):
            n_in_units=layers[l]
            n_out_units=layers[l+1]

            moduleLayers.append(nn.Linear(n_in_units,n_out_units))
            #apply the activation function for all layers except the last
            #(latent) layer 
            act_fct= nn.Identity() if l==len(layers)-2 else self.activation_fct
            moduleLayers.append(act_fct)

        sequential=nn.Sequential(*moduleLayers)
        return sequential

    def decode(self, posterior_sample):
        logger.debug("Decoder::decode")
        return self._network(posterior_sample)
        
    
    def get_activation_fct(self):        
        return

if __name__=="__main__":
    logger.debug("Testing Networks")
    nw=Network()
    encoder=SimpleEncoder()
    logger.debug(encoder._layers)
    decoder=SimpleDecoder()
    logger.debug(decoder._layers)
    hierarchicalEncoder=HierarchicalEncoder()
    decoder2=Decoder()
    # prior=Prior(create_module_list=True)
    # logger.debug(prior._layers)
    logger.debug("Success")
    pass