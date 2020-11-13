import torch

import logging
logger = logging.getLogger(__name__)

def generate_iterative_samples(model):
    num_latent_units=100
    #TODO the range of this is taken from the clamping of the posterior
    #samples to -88,88. Where is this coming from? Check the values
    #TODO check if this is actually the right sampling procedure...
    z0=-166*torch.rand([num_latent_units])+88
    z1=-166*torch.rand([num_latent_units])+88
    z2=-166*torch.rand([num_latent_units])+88
    z3=-166*torch.rand([num_latent_units])+88

    init_samples_left=torch.cat([z0,z1])
    init_samples_right=torch.cat([z2,z3])		

    output=model.generate_samples_per_gibbs(init_left_samples=init_samples_left, init_right_samples=init_samples_right,steps=10)
    # print(output)
    out_tensor=torch.cat(output)
    out_tensor=out_tensor.detach()
    from helpers import plot_rbm_output
    plot_rbm_output(out_tensor, n_samples=50, output="./output/divae_mnist/rbm_samples/rbm_sampling_2011110_successive.png")
    return

def generate_samples(model):
    output=model.generate_samples(n_samples=100)
    output=output.detach()
    from helpers import plot_rbm_output
    plot_rbm_output(output, n_samples=100, output="./output/divae_mnist/rbm_samples/rbm_sampling_2011110.png")
    return

def generate_samples_vae(model):
    output=model.generate_samples(n_samples=100)
    output=output.detach()
    from helpers import plot_rbm_output
    plot_rbm_output(output, n_samples=100, output="./output/divae_mnist/rbm_samples/rbm_sampling_2011110.png")
    return


if __name__=="__main__":
    logger.info("Testing RBM Setup")

    # BATCH_SIZE = 32
    # VISIBLE_UNITS = 784  # 28 x 28 images
    # HIDDEN_UNITS = 128
    # N_GIBBS_SAMPLING_STEPS = 10
    # EPOCHS = 6
    # N_EVENTS_TRAIN=-1
    # N_EVENTS_TEST=-1
    # do_train=False
    # config_string="_".join(map(str,[N_EVENTS_TRAIN,EPOCHS,N_GIBBS_SAMPLING_STEPS]))

    # from data.loadMNIST import loadMNIST
    # train_loader,test_loader=loadMNIST(
    # 		batch_size=BATCH_SIZE,
    # 		num_evts_train=N_EVENTS_TRAIN,
    # 		num_evts_test=N_EVENTS_TEST,
    # 		binarise="threshold")
    
    # rbm = RBM(n_visible=VISIBLE_UNITS, n_hidden=HIDDEN_UNITS, n_gibbs_sampling_steps=N_GIBBS_SAMPLING_STEPS)

    # f=open("./output/divae_mnist/model_DiVAE_mnist_500_-1_100_1_0.001_4_100_RELU_default_201104.pt",'rb')
    # model=torch.load(f)
    # print(model)
    # f.close()
    # # ########## EXTRACT FEATURES ##########
    # logger.info("Sampling from RBM")
    # for batch_idx, (x_true, label) in enumerate(test_loader):
    # 	y=rbm.get_samples(x_true.view(-1,VISIBLE_UNITS))
    # 	break
    # from helpers import plot_MNIST_output

    # plot_MNIST_output(x_true,y, n_samples=5, output="./output/rbm_test_200827_wdecay_{0}.png".format(config_string))
