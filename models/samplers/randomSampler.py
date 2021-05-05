"""
Sampler returning random numbers
"""

from models.samplers.baseSampler import BaseSampler

class RandomSampler(BaseSampler):
    def __init__(self, n_gibbs_sampling_steps=0, **kwargs):
        super(RandomSampler, self).__init__(**kwargs)

    def get_samples(self, n_latent_nodes=100, n_latent_hierarchy_lvls=4,):
        ##Sampling mode: random (no sampling)
        # flat, uniform sampled z, no dependence. Straight to decoder.
        # good as reference to compare random numbers to RBM samples

        rnd_samples=[]
        for i in range(0,n_latent_hierarchy_lvls):
            #TODO the range of this is taken from the clamping of the posterior
            #samples to -88,88. Where is this coming from? Check the values again.
            rnd_samples.append(-166*torch.rand([n_latent_nodes])+88)
        return rnd_samples
