"""
Simple encoder

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
from networks.network import Network
import torch

class SimpleEncoder(Network):
    def __init__(self,smoothing_distribution=None, n_latent_hierarchy_lvls=4, **kwargs):
        super(SimpleEncoder, self).__init__(**kwargs)
        self.smoothing_distribution=smoothing_distribution

        #number of hierarchy levels in encoder. This is the number of latent
        #layers. At each hiearchy level an output layer is formed.
        self.n_latent_hierarchy_lvls=4
        #number of latent nodes in the prior - output nodes for each level of
        #the hierarchy. Also number of input nodes to the decoder, first layer
        self.n_latent_nodes=100
        #each hierarchy has NN with n_encoder_layers_enc layers
        #number of deterministic nodes in each encoding layer. These layers map
        #input to the latent layer. 
        self.n_encoder_layer_nodes=200
        # number of deterministic layers in each conditional p(z_i | z_{k<i})
        self.n_encoder_layers=2 

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
            to n_latent_hierarchy_lvls and each element in the list is a DistUtil object containing posterior distribution 
            for the group of latent nodes in each hierarchy level. 

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
        for i in range(self.n_latent_hierarchy_lvls):
            qprime=self.encode(x)
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