import os
import torch
import numpy as np

class Wassterstein(object):
    def __init__(self):
        self.gen_loss = []
        self.disc_loss = []
    def update_single_wasserstein(self, X, Y, src_lab, targ_lab):
        #scale = torch.cuda.FloatTensor([0.4, 0.6])
        batch_size = X.shape[0]
        wasserstein_distance_src = 0
        wasserstein_distance_targ = 0
        for bat in range(batch_size):
            #WD = (X[bat,:,:].reshape(X.shape[1]*X.shape[2])).sum()/(torch.sum(src_lab[bat, :, :])+1e-7) \
            #      - (Y[bat,:,:].reshape(Y.shape[1]*Y.shape[2])).sum()/(torch.sum(targ_lab[bat, :, :]) + 1e-7)
            #wasserstein_distance = torch.mul(wasserstein_distance, scale)
            WD =  X[bat,:,:].reshape(X.shape[1]*X.shape[2]).mean()
            wasserstein_distance_src = wasserstein_distance_src + WD
            WD =  Y[bat,:,:].reshape(Y.shape[1]*Y.shape[2]).mean()
            wasserstein_distance_targ = wasserstein_distance_targ + WD
        #print(wasserstein_distance)
        return wasserstein_distance_src/batch_size, wasserstein_distance_targ/batch_size

    def update_wasserstein(self, X, Y, src_lab, targ_lab):
        #scale = torch.cuda.FloatTensor([0.4, 0.6])
        batch_size = X.shape[0]
        wasserstein_distance_source = 0
        wasserstein_distance_target = 0
        import pdb
        for bat in range(batch_size):
            WD_s = ((X[bat,:,:] * src_lab[bat]).reshape(X.shape[1]*X.shape[2])).sum()/(torch.sum(src_lab[bat, :, :])+1e-7)
            WD_t =  ((Y[bat, :,:] * targ_lab[bat]).reshape(Y.shape[1]*Y.shape[2])).sum()/(torch.sum(targ_lab[bat, :, :]) + 1e-7)
            #wasserstein_distance = torch.mul(wasserstein_distance, scale)
            #WD =  X[bat,:,:].reshape(X.shape[1]*X.shape[2]).mean()  - Y[bat,:,:].reshape(Y.shape[1]*X.shape[2]).mean()
            wasserstein_distance_source = wasserstein_distance_source + WD_s
            wasserstein_distance_target = wasserstein_distance_target + WD_t
        #print(wasserstein_distance)
        return wasserstein_distance_source/batch_size, wasserstein_distance_target/batch_size



    def gradient_regularization(self, critic, h_s, h_t):
        import pdb
        alpha = torch.rand(h_s.size(0),1).cuda()
        #pdb.set_trace()
        alpha = alpha.expand(h_s.size(0), int(h_s.nelement()/h_s.size(0))).contiguous().view(h_s.size(0), h_s.size(1), h_s.size(2), h_s.size(3))
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        interpolates = interpolates.cuda()
        #interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        _, preds = critic(interpolates)
        gradients = torch.autograd.grad(preds, interpolates,
                        grad_outputs=torch.ones_like(preds),
                        retain_graph=True, create_graph=True)[0]
        penalty_cup = 0
        penalty_disc = 0
        gradients_cup = gradients[:, 1, :,:]
        gradients_disc = gradients[:, 1, :,:]

        gradients_ = gradients_cup.view(2, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients_ ** 2, dim=1) + 1e-12)
        penalty_cup= (torch.max(torch.zeros(1).float().cuda(), (gradient_norm - 1))**2).mean()

        gradients_ = gradients_disc.view(2, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients_ ** 2, dim=1) + 1e-12)
        penalty_disc= (torch.max(torch.zeros(1).float().cuda(), (gradient_norm - 1))**2).mean()
        return penalty_cup, penalty_disc
    def gradient_penalty_(self, critic, h_s, h_t):
        # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
        import pdb
        alpha = torch.rand(h_s.size(0),1).cuda()
        #pdb.set_trace()
        alpha = alpha.expand(h_s.size(0), int(h_s.nelement()/h_s.size(0))).contiguous().view(h_s.size(0), h_s.size(1), h_s.size(2), h_s.size(3))
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        interpolates = interpolates.cuda()
        #interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        _, preds = critic(interpolates)
        gradients = torch.autograd.grad(preds, interpolates,
                        grad_outputs=torch.ones_like(preds),
                        retain_graph=True, create_graph=True)[0]
        ###############TO BE CHECKED AND UPDATED################!!
        import pdb
        #pdb.set_trace()
        gradient_penalty = 0
        gradient_penalty_disc = 0
        for i,_ in enumerate(['disc', 'cup']):
            gradients_ = gradients[:,i,:,:]
            gradients_ = gradients_.view(1, -1)
            #print('Gradients', gradients)
            gradient_norm = torch.sqrt(torch.sum(gradients_ ** 2, dim=1) + 1e-12)
            #print('Gradients_norm', gradients)
            #gradient_norm = gradients.norm(2, dim=-1)
            gradient_penalty += ((gradient_norm - 1)**2).mean()
        return gradient_penalty

    # def gradient_penalty(self, D, real_samples, fake_samples):
    #     """Calculates the gradient penalty loss for WGAN GP"""
    #     # Random weight term for interpolation between real and fake samples
    #     alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    #     # Get random interpolation between real and fake samples
    #     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    #     _, d_interpolates = D(interpolates)
    #     fake = torch.autograd.Variable(torch.cuda.FloatTensor(torch.ones_like(d_interpolates)), requires_grad=False)
    #     # Get gradient w.r.t. interpolates
    #     gradients = torch.autograd.grad(
    #         outputs=d_interpolates,
    #         inputs=interpolates,
    #         grad_outputs=fake,
    #         create_graph=True,
    #         retain_graph=True
    #     )[0]
    #     gradients = gradients.view(gradients.size(0), -1)
    #     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #     return gradient_penalty