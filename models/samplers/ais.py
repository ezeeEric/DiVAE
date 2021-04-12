"""
Annealed Importance Sampling for approximating the free energy of an RBM.

Adapted from: https://pydeep.readthedocs.io/en/latest/_modules/pydeep/rbm/estimator.html?highlight=pydeep.rbm.estimator
"""

import numpy as np
import torch
from torch import nn
from copy import deepcopy

from models.rbm.rbm import RBM
from models.rbm.rbmBase import RBMBase
from models.samplers.baseSampler import BaseSampler

from DiVAE import logging
logger = logging.getLogger(__name__)

class AnnealedImportanceSampler(BaseSampler):
    def __init__(self, target_rbm=None, n_betas=10000, n_ais_chains=1000, **kwargs):
        super(AnnealedImportanceSampler, self).__init__(**kwargs)
        #number of inverse temperatures
        self._n_betas = n_betas

        #number of simultaneous AIS chains
        self._n_ais_chains = n_ais_chains

        #List of n_beta evenly spaced numbers over [0,1] - inverse temps
        self._betas=np.linspace(0.0,1.0,self._n_betas)
    
        #the target
        self._target_rbm=target_rbm

        self._n_visible = self._target_rbm._n_visible
        self._n_hidden  = self._target_rbm._n_hidden

        #the proposal distribution/base-rate RBM
        self._base_rbm = self.create_base_model()


    def create_base_model(self):
        #This base rate RBM A is the starting point o0f the annealing schedule.
        #Arbitrarily, the Weight matrix is set to 0 and the hidden biases are
        #set to 0 too. #TODO not sure the hidden biases should be 0. Weights yes.
        # self._base_rbm=RBM(self._target_rbm._n_visible,self._target_rbm._n_hidden)
        rbm=RBMBase(n_visible=self._n_visible,n_hidden=self._n_hidden)
        #zero weight matrix
        rbm.weights = torch.zeros(self._target_rbm.weights.size())
        #TODO initialise to best working value...
        rbm.visible_bias = self._target_rbm.visible_bias
        #TODO should/could this be 0? What's the impact on the calculation
        rbm.hidden_bias = self._target_rbm.hidden_bias

        # rbm.visible_bias=0.5*torch.ones(self._n_visible)
        # rbm.hidden_bias=torch.zeros(self._n_hidden)
        return rbm

#https://pydeep.readthedocs.io/en/latest/_modules/pydeep/rbm/model.html
    def unnorm_log_prob_v(self, vis, beta=None):
        #(1-beta)*bA*v
        lnpv_1=torch.sum((1-beta)*self._base_rbm.visible_bias*vis,1).reshape(self._n_ais_chains,1)
        
        lnpv_2=0 #assume MA=0/aj=0
        
        #beta*bB*v
        lnpv_3=beta*torch.sum(self._target_rbm.visible_bias*vis,1).reshape(self._n_ais_chains,1)
        
        #sum hid(log(1+exp(beta*(wB*v+aB)))
        #TODO check if this is all correctly multiplied.
        act_hid=torch.matmul(vis,self._target_rbm.weights)+self._target_rbm.hidden_bias
        lnpv_3=torch.sum(torch.log(1+torch.exp(beta*act_hid)),1).reshape(self._n_ais_chains,1)
        
        lnpv=lnpv_1+lnpv_2+lnpv_3
        
        return lnpv.detach()

    def get_logZ_base_model(self):
        #M*log 2+sum(log(1+exp(vis_b)))
        import math
        logZ=self._n_hidden*math.log(2)+torch.sum(torch.log(1+torch.exp(self._base_rbm.visible_bias)))
        return logZ

    def sample(self):
        ############# 1.
        # The base model is the RBM with 0 weights, only the biases contribute to
        # the partition fct.
        # Sample the first time from the base model
        out_fct=nn.Sigmoid()
        base_rbm=self.create_base_model()

        #p(v|h) base model = sig(vis bias)
        #because Wij=0
        p_v_init=out_fct(self._base_rbm.visible_bias)
        #this size is determined by the number of AIS chains M we want to run
        #(see DBN paper "On the Quantitative Analysis of Deep Belief Networks")
        #and the size of the RBM visible layer - we want M v^(i) samples and
        #each v^(i) = n_visible nodes of RBM
        v_size=(self._n_ais_chains,p_v_init.size()[0])
        rnd_nr=torch.rand(v_size) 
        #initial sampling step: create M v_0 from base distribution RBM_A  
        v_sampled=torch.where(rnd_nr>=p_v_init,1.,0.)
        ############# 2.
        #initial base rate pA: 
        #negative as this is the divisor of 
        logpvsum=-torch.sum(self._base_rbm.visible_bias*v_sampled,1).reshape(self._n_ais_chains,1)
        
        ############# 3.
        for beta in self._betas[1:][:-1]:
            logpvsum+=self.unnorm_log_prob_v(v_sampled, beta=beta)
            ############# 4
            # Gibbs Sampler
            for step in range(self._n_gibbs_sampling_steps):
                #sample h 
                hid_act=beta * torch.matmul(v_sampled,self._target_rbm.weights)+self._target_rbm.hidden_bias
                # print(hid_act.size())
                # hid_act=hid_act.reshape(self._n_ais_chains,1)
                phv=out_fct(hid_act)
                rnd_nr=torch.rand(phv.size()) 
                h_sampled=torch.where(rnd_nr>=phv,1.,0.)

                b_vis=(1-beta)*self._base_rbm.visible_bias+self._target_rbm.visible_bias*beta
                vis_act=beta*torch.matmul(h_sampled,self._target_rbm.weights)+b_vis
                pvh=out_fct(vis_act)
                v_sampled=torch.where(rnd_nr>=pvh,1.,0.)
            
            logpvsum-=self.unnorm_log_prob_v(v_sampled, beta=beta)
        beta_K=self._betas[len(self._betas)-1]
        logpvsum+=self.unnorm_log_prob_v(v_sampled, beta=beta_K)

        #combine all simultaneous AIS chains
        #TODO check this
        logz=torch.logsumexp(logpvsum,0)-np.log(self._n_ais_chains)
        baselogz = self.get_logZ_base_model()
        # Add the base partition function
        logz = logz + baselogz


        return logz
        # +/- 3 standard deviations
        # logpvmean = torch.mean(logpvsum)
        # print(logpvmean)
        exit()
        # logpvstd = torch.log(numx.std(numx.exp(lnpvsum - lnpvmean))) + lnpvmean - numx.log(num_chains) / 2.0
        # lnpvstd = numx.vstack((numx.log(3.0) + lnpvstd, logz))

        # Calculate partition function of base distribution
        baselogz = model._base_log_partition(True)
    
        # Add the base partition function
        logz = logz + baselogz
        logz_up = numxext.log_sum_exp(lnpvstd) + baselogz
        logz_down = numxext.log_diff_exp(lnpvstd) + baselogz

# def log_diff_exp(x, axis=0):
#     """ Calculates the logarithm of the diffs of e to the power of input 'x'. The method tries to avoid \
#         overflows by using the relationship: log(diff(exp(x))) = alpha + log(diff(exp(x-alpha))).

#     :param x: data.
#     :type x: float or numpy array

#     :param axis: Diffs along the given axis.
#     :type axis: int

#     :return: Logarithm of the diff of exp of x.
#     :rtype: float or numpy array.
#     """
#     alpha = x.max(axis) - numx.log(numx.finfo(numx.float64).max) / 2.0
#     if axis == 1:
#         return numx.squeeze(alpha + numx.log(numx.diff(numx.exp(x.T - alpha), n=1, axis=0)))
#     else:
#         return numx.squeeze(alpha + numx.log(numx.diff(numx.exp(x - alpha), n=1, axis=0)))

        print(logz)
        exit()
        ############# 5
        # TODO the math suggests there might be additional terms here for the
        # hidden biases.        
        # logpvsum=torch.sum((1-beta)*self._base_rbm.visible_bias*vis,1)
        # print(logpvsum)
        # unnormalized_log_probability_v
        # sum_logpv = - torch.dot(v_sampled,self._base_rbm.visible_bias)
        # print("Initial sum log p_v",sum_logpv)
        
        # #exclude 0 and 1 #TODO
        #     # print(beta)

if __name__=="__main__": 
    logger.info("Loading Model")
    # rbm=RBM(n_visible=200,n_hidden=200)
    # input_rbm="/Users/drdre/codez/qVAE/DiVAE/outputs/2021-03-17/10-54-18/rbm.pt"
    # rbm.load_state_dict(torch.load(input_rbm))
    from models.rbm.exactRBMPartFctSolver import ExactPartitionSolver
    import pickle
    f=open("/Users/drdre/codez/qVAE/DiVAE/output/210324_eps/rbm_1010.pkl",'rb')
    # Exact Partition Fct for RBM(10,10)
    # Z=4886438400.0
    # logZ=22.309728622436523
    eps=pickle.load(f)
    Z=pickle.load(f)
    logZ=pickle.load(f)
    n_visible=eps._rbm_visible_bias.size()[0]
    n_hidden=eps._rbm_hidden_bias.size()[0]
    rbm=RBM(n_visible=n_visible,n_hidden=n_hidden)

    rbm._weights=torch.nn.Parameter(eps._rbm_weights,requires_grad=False)
    rbm._visible_bias=torch.nn.Parameter(eps._rbm_visible_bias,requires_grad=False)
    rbm._hidden_bias=torch.nn.Parameter(eps._rbm_hidden_bias,requires_grad=False)
    
    ais=AnnealedImportanceSampler(rbm=rbm, n_gibbs_sampling_steps=10)
    ais.sample()
    # n=2
    # x=int(2**n)
    # print(binary(torch.IntTensor(x),n))
    # # print(bin(x))
