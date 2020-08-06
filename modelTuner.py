import torch
import numpy as np

"""Methods for training and testing."""

def trainDiVAE(model, train_loader, optimizer, epoch):
    print("start train()")
    model.train()
    total_train_loss = 0
    for batch_idx, (x_true, label) in enumerate(train_loader):
        optimizer.zero_grad()
        x_true = torch.autograd.Variable(x_true)
        x_recon, posterior_distribution, posterior_samples = model(x_true)
        train_loss = model.lossDiVAE(x_true, x_recon, posterior_distribution, posterior_samples)
        import sys
        sys.exit()
        train_loss.backward()
        total_train_loss += train_loss.item()
        optimizer.step()
        
        # Output logging
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x_true), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.data.item() / len(x_true)))
    print("finish train()")
    return total_train_loss/len(train_loader.dataset)

def testDiVAE(model, test_loader):
    print("call test()")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x_true, label) in enumerate(test_loader):
            x_recon, mu, logvar = model(x_true)
            test_loss += model.loss(x_true, x_recon, mu, logvar)
        
    test_loss /= len(test_loader.dataset)
    print(test_loss)
    print("finished test()")
    return test_loss

def evaluateDiVAE(model, test_loader, batch_size=128, latent_dimensions=32):
    print("call evaluate()")
    batch_mu = np.zeros((batch_size, latent_dimensions))
    batch_logvar = np.zeros((batch_size, latent_dimensions))

    with torch.no_grad():
        for batch_idx, (x_true, label) in enumerate(test_loader):
            x_recon, batch_mu, batch_logvar = model(x_true)

    print("call finish()")
    return x_true, x_recon

def train(model, train_loader, optimizer, epoch):
    print("start train()")
    model.train()
    total_train_loss = 0
    for batch_idx, (x_true, label) in enumerate(train_loader):
        optimizer.zero_grad()
        x_true = torch.autograd.Variable(x_true)
        if model.type=='AE':
            x_recon = model(x_true)
            train_loss = model.loss(x_true,x_recon)
        elif model.type=='VAE':
            x_recon, mu, logvar = model(x_true)
            train_loss = model.loss(x_true, x_recon, mu, logvar)
        elif model.type=='DiVAE':
            x_recon, mu, logvar = model(x_true)
            train_loss = model.loss(x_true, x_recon, mu, logvar)     
        train_loss.backward()
        total_train_loss += train_loss.item()
        optimizer.step()
        
        # Output logging
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x_true), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.data.item() / len(x_true)))
    print("finish train()")
    return total_train_loss/len(train_loader.dataset)

def test(model, test_loader):
    print("call test()")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x_true, label) in enumerate(test_loader):
            if model.type=='AE':
                x_recon = model(x_true)
                test_loss = model.loss(x_true,x_recon)
            elif model.type=='VAE':
                x_recon, mu, logvar = model(x_true)
                test_loss = model.loss(x_true, x_recon, mu, logvar)
            elif model.type=='DiVAE':
                x_recon, mu, logvar = model(x_true)
                test_loss = model.loss(x_true, x_recon, mu, logvar) 
        
    test_loss /= len(test_loader.dataset)
    print(test_loss)
    print("finished test()")
    return test_loss

def evaluate(model, test_loader, batch_size=128, latent_dimensions=32):
    print("call evaluate()")
    batch_mu = np.zeros((batch_size, latent_dimensions))
    batch_logvar = np.zeros((batch_size, latent_dimensions))

    with torch.no_grad():
        for batch_idx, (x_true, label) in enumerate(test_loader):
            if model.type=='AE':
                x_recon = model(x_true)
            elif model.type=='VAE':
                x_recon, mu, logvar = model(x_true)
            elif model.type=='DiVAE':
                x_recon, mu, logvar = model(x_true)

    print("call finish()")
    return x_true, x_recon

