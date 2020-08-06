import torch
import torch.distributions as dist

import smoothers

#sample 0 or 1 with probability x from Bernoulli
means=[0.9,0.8,0.1]

p=dist.Bernoulli(torch.tensor([means[0]]))
q=dist.Bernoulli(torch.tensor([means[1]]))
r=dist.Bernoulli(torch.tensor([means[2]]))

print(p.log_prob(0))
# for i in range(0,10):
#     print(p.sample())
#     print(q.sample())

divKL_pq=dist.kl.kl_divergence(p,q)
divKL_pr=dist.kl.kl_divergence(p,r)
print("KL-Div between Bernoulli with means {0} and {1}: {2}".format(str(means[0]),str(means[1]),divKL_pq))
print("KL-Div between Bernoulli with means {0} and {1}: {2}".format(str(means[1]),str(means[2]),divKL_pr))

# spexSmoother=smoothers.SpikeAndExponentialSmoother()
# object_methods = dir(spexSmoother)
# for method in object_methods:
#     print(method)
# spexSmoother.sample()