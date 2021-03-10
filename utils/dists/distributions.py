

"""
distributions.py

A set of classes representing the smoothing functions that are used as a bridge
from discrete to continuous variables when training a DVAE.

Author :  ODM, Dr. Dre
"""

import torch
from torch import zeros, ones
from torch.distributions import Distribution, Uniform

import numpy as np
import logging
logger = logging.getLogger(__name__)

def sigmoid_cross_entropy_with_logits(logits, labels):
        # this is the equivalent to the tensorflow
        # sigmoid_cross_entropy_with_logits(): 
        # return logits * labels + tf.nn.softplus(-logits)
        #TODO cross check this implementation
        sp=torch.nn.Softplus()
        return logits - logits * labels + sp(-logits)


class Bernoulli(Distribution):
    def __init__(self, logits=None,  beta=1,  **kwargs):
        super(Bernoulli, self).__init__(**kwargs)
        #this is the raw (no output fct) output data of the latent layer stored
        #in this distribution.
        assert logit is not None, 'Distributions must be initialised with logit'
        assert not beta<=0, 'beta larger 0'
        self.logits = logits
        self.beta = beta

    def reparameterise(self):
        #draw samples from bernoulli distribution with probability p=1-q
        #where q is logits and rho acts as sample 
        #returns 0/1
        q = torch.sigmoid(self.logits) # equals 1-probability
        rho = torch.rand(q.size())
        bernoulli_sample = torch.where(rho<q, torch.ones(q.size()), zeros(q.size()))
        return bernoulli_sample
    
    def entropy(self):
        """
        Computes the entropy of the bernoulli distribution using:
            x - x * z + log(1 + exp(-x)),  where x is logits, and z=sigmoid(x).
        Returns: 
            ent: entropy
        """
        q = torch.sigmoid(self.logits)
        entropy = sigmoid_cross_entropy_with_logits(logits=self.logits, labels=x)
        return entropy

    def log_prob_per_var(self, samples):
        """
        Compute the log probability of samples under distribution of this object.
            - (x - x * z + log(1 + exp(-x))),  where x is logits, and z is samples.
        Args:
            samples: matrix of size (num_samples * num_vars)

        Returns: 
            log_prob: a matrix of log_prob (num_samples * num_vars).
        """
        log_prob = - sigmoid_cross_entropy_with_logits(logits=self.logits, labels=samples)
        return log_prob

    def __repr__(self):
        return "\n".join([str(item) for item in self.__dict__.items()])

class SpikeAndExponentialSmoother(Bernoulli):
    """ 
    Spike-and-exponential smoother from the original DVAE paper of Rolfe.
    """
    def __init__(self,**kwargs):
        super(SpikeAndExponentialSmoother, self).__init__(**kwargs)

    def reparameterise(self):
        #TODO cross check this
        #this is the approximate posterior probability
        q = torch.sigmoid(self.logits)
        #clip the probabilities 
        q = torch.clamp(q,min=1e-7,max=1.-1e-7)
        #this is a tensor of uniformly sampled random number in [0,1)
        rho=torch.rand(q.size())
        zero_mask = zeros(q.size())
        ones=torch.ones(q.size())
        #calculate ICDF of SpikeAndExponential
        interior_log = ((rho+q-ones)/q)*(np.exp(self.beta)-1)+ones
        conditional_log = (1./self.beta)*torch.log(interior_log)
        zeta=torch.where(rho >= 1 - q, conditional_log, zero_mask)
        return zeta

    def entropy(self,x):
        """Computes the entropy of the bernoulli distribution using:
            x - x * z + log(1 + exp(-x)),  where x is logits, and z=sigmoid(x).
        Returns: 
            ent: entropy
        """
        z  = torch.sigmoid(x)
        ent=x-x*z+torch.log(1+torch.exp(-x))
        return ent

def visualise_distributions(rho,q,samples):
    import matplotlib.pyplot as plt
    samples=torch.flatten(samples)
    plt.hist(samples.detach().numpy(),bins=100)
    plt.show()

if __name__=="__main__":
    logger.info("Testing DistUtils Setup") 
    smooth=SpikeAndExponentialSmoother()
    import pickle
    qprime=pickle.load(open("datasample.pkl","rb"))
    rho=torch.rand(qprime.size())
    sigmoid=torch.nn.Sigmoid()
    q=sigmoid(qprime)
    rho=torch.rand(q.size())
    samples=smooth.icdf(rho,q)
    visualise_distributions(rho,q,samples)
    logger.info("Success")
    pass