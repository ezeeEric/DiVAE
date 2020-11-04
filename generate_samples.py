import torch

import logging
logger = logging.getLogger(__name__)

def generate_samples(model):
    # print(model)
    output=model.generate_samples(n_samples=100)
    output=output.detach()
    from helpers import plot_rbm_output
    plot_rbm_output(output, n_samples=100, output="./output/divae_mnist/rbm_sampling_test_nogibbs.png")
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
