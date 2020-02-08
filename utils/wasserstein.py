import os
import torch
import numpy as np
import torch.distributions.dirichlet as dirichlet

class Wasserstein(object):
    def __init__(self):
        self.gen_loss = []
        self.disc_loss = []
    def update_single_wasserstein(self, X, Y):
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

    def update_wasserstein_dual_source(self, X, Y):
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
        wass_loss = wasserstein_distance_src/batch_size - wasserstein_distance_targ/batch_size
        return wass_loss

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


    def gradient_regularization_dual_source(self, critic, h_s, h_t, batch_size, num_source):
        alpha1, alpha2, alpha3 = self.dirichlet_number_generator(batch_size * num_source)
        source1 = h_s[0:batch_size]
        source2 = h_s[batch_size:num_source*batch_size]
        target = h_t[0:batch_size]
        import pdb
        pdb.set_trace()
        alpha1 = alpha1.expand(source1.size(0), int(source1.nelement()/source1.size(0))).contiguous().view(source1.size(0), source1.size(1), source1.size(2), source1.size(3)).cuda()
        alpha2 = alpha2.expand(source2.size(0), int(source2.nelement()/source2.size(0))).contiguous().view(source2.size(0), source2.size(1), source2.size(2), source2.size(3)).cuda()
        alpha3 = alpha3.expand(target.size(0), int(target.nelement()/target.size(0))).contiguous().view(target.size(0), target.size(1), target.size(2), target.size(3)).cuda()
        # alpha 1 = batch_size  * num_labels * H * W
        interpolates = torch.cat([alpha3 * target, alpha3 * target])  + torch.cat([alpha1 * source1 ,alpha2 * source2])
        interpolates = interpolates.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        _, preds = critic(interpolates) # size batch_size * num_src_domains
        gradients = torch.autograd.grad(preds, interpolates,
                        grad_outputs=torch.ones_like(preds),
                        retain_graph=True, create_graph=True)[0]
        penalty_cup = 0
        penalty_disc = 0
        gradients_cup = gradients[:, 1, :,:]
        gradients_disc = gradients[:, 0, :,:]

        gradients_ = gradients_cup.view(2, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients_ ** 2, dim=1) + 1e-12)
        penalty_cup= (torch.max(torch.zeros(1).float().cuda(), (gradient_norm - 1))**2).mean()

        gradients_ = gradients_disc.view(2, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients_ ** 2, dim=1) + 1e-12)
        penalty_disc= (torch.max(torch.zeros(1).float().cuda(), (gradient_norm - 1))**2).mean()
        return penalty_cup, penalty_disc

    def gradient_regularization(self, critic, h_s, h_t):
        alpha = torch.rand(h_s.size(0),1).cuda()
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
        gradients_disc = gradients[:, 0, :,:]

        gradients_ = gradients_cup.view(2, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients_ ** 2, dim=1) + 1e-12)
        penalty_cup= (torch.max(torch.zeros(1).float().cuda(), (gradient_norm - 1))**2).mean()

        gradients_ = gradients_disc.view(2, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients_ ** 2, dim=1) + 1e-12)
        penalty_disc= (torch.max(torch.zeros(1).float().cuda(), (gradient_norm - 1))**2).mean()
        return penalty_cup, penalty_disc

    def gradient_penalty_(self, critic, h_s, h_t):
        # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
        if len(h_s) == 2:
            alpha1, alpha2 = dirichlet_number_generator()
            alpha1 = alpha1.expand(h_s[0].size(0), int(h_s[0].nelement()/h_s[0].size(0))).contiguous().view(h_s[0].size(0), h_s[0].size(1), h_s[0].size(2), h_s[0].size(3))
            alpha2 = alpha2.expand(h_s[1].size(0), int(h_s[1].nelement()/h_s[1].size(0))).contiguous().view(h_s[1].size(0), h_s[1].size(1), h_s[1].size(2), h_s[1].size(3))
            differences = h_t - h_s
            interpolates = h_s + (alpha * differences)
        else:
            alpha = torch.rand(h_s.size(0),1).cuda()
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

    def dirichlet_number_generator(self, total_size):
        m = dirichlet.Dirichlet(torch.Tensor([total_size, total_size, total_size]))
        return m.sample()