from utils.loss import SegmentationLosses
from models.deeplab import *
import torch.nn.functional as F
from utils.lr_scheduler import LR_Scheduler
from models.sync_batchnorm.replicate import patch_replication_callback
from models.discriminator import FCDiscriminator, FCDiscriminator_WGAN
from utils.wasserstein import Wasserstein
import torch
import os
from torch import nn
import pdb

class mwdan_trainer(object):
    def __init__(self, args, nnclass, ndomains):
        self.device = 1
        self.generator_model = None
        self.generator_optim = None
        self.generator_criterion = None
        self.batch_size = args.batch_size
        self.nnclass = nnclass
        self.num_domains = ndomains
        self.init_wasserstein = Wasserstein()
        self.init_generator(args)
        self.init_discriminator(args)
        self.init_optimizer(args)

    def init_generator(self, args):

        self.generator_model = DeepLab(num_classes=self.nnclass,
                            backbone='resnet',
                            output_stride=16,
                            sync_bn = None,
                            freeze_bn=False).cuda()

        self.generator_model = torch.nn.DataParallel(self.generator_model).cuda()
        patch_replication_callback(self.generator_model)
        if args.resume:
            print('#--------- load pretrained model --------------#')
            model_dict = self.generator_model.module.state_dict()
            checkpoint = torch.load(args.resume)
            pretrained_dict = {k:v for k,v in checkpoint['state_dict'].items() if 'last_conv' not in k and k in model_dict.keys() }
            #pretrained_dict = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()  if 'last_conv' not in k}
            model_dict.update(pretrained_dict)
            self.generator_model.module.load_state_dict(model_dict)
        for param in self.generator_model.parameters():
            param.requires_grad = True

    def init_discriminator(self, args):
        # init D
        self.discriminator_model = FCDiscriminator(num_classes=2).cuda()
        self.interp = nn.Upsample(size=400, mode='bilinear')
        self.disc_criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        return

    def init_optimizer(self, args):
        self.generator_criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='bce') #torch.nn.BCELoss(reduce ='mean')
        self.generator_params = [{'params': self.generator_model.module.get_1x_lr_params(), 'lr':args.lr},
                                {'params': self.generator_model.module.get_10x_lr_params(), 'lr': args.lr * 10}]
        self.discriminator_params=[{'params': self.discriminator_model.parameters(), 'lr': args.lr*5}]
        self.model_optim = torch.optim.Adadelta(self.generator_params+self.discriminator_params)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, lr_step=30, iters_per_epoch=100)

    # for madan the src_image has shape B x source_index x channel x H x W
    def update_weights(self, src_image, src_labels, tar_image, tar_labels,options):
        running_loss = 0.0
        src_labels = torch.cat([src_labels[:,0].squeeze(), src_labels[:,1].squeeze()], 0).type(torch.LongTensor).cuda()
        self.model_optim.zero_grad()
        # src image shape batch_size x domain x 3 channels x height x width
        src_out, source_feature = self.generator_model(torch.cat([src_image[:,0].squeeze(), src_image[:,1].squeeze()]))
        tar_out, target_feature = self.generator_model(tar_image)
        #  Discriminator
        discriminator_x = torch.cat([source_feature, target_feature]).squeeze()
        disc_clf = self.discriminator_model(discriminator_x)
        # Losses
        losses = torch.stack([self.generator_criterion(src_out[j*self.batch_size:j+self.batch_size], src_labels[j*self.batch_size:j+self.batch_size]) for j in range(self.num_domains)])
        slabels = torch.ones(self.batch_size, disc_clf.shape[2], disc_clf.shape[3], requires_grad=False).type(torch.LongTensor).cuda()
        tlabels = torch.zeros(self.batch_size*2, disc_clf.shape[2], disc_clf.shape[3], requires_grad=False).type(torch.LongTensor).cuda()
        domain_losses = torch.stack([self.generator_criterion(disc_clf[j*self.batch_size:j+self.batch_size].squeeze(), slabels) for j in range(self.num_domains)])
        domain_losses = torch.cat([domain_losses, self.generator_criterion(disc_clf[2*self.batch_size:2*self.batch_size+2*self.batch_size].squeeze(), tlabels).view(-1)])
        # Different final loss function depending on different training modes.
        if options['mode']== "maxmin":
            loss = torch.max(losses) + options['mu'] * torch.min(domain_losses)
        elif options['mode'] == "dynamic":
            loss = torch.log(torch.sum(torch.exp(options['gamma'] * (losses + options['mu'] * domain_losses)))) / options['gamma']
        else:
            raise ValueError("No support for the training mode on madnNet: {}.".format(options['mode']))
        loss.backward()
        self.model_optim.step()
        running_loss += loss.detach().cpu().numpy()
        # compute target loss
        target_loss = self.generator_criterion(tar_out, tar_labels).detach().cpu().numpy()
        return running_loss, target_loss

    def update_wasserstein(self, srca_image, srca_labels, srcb_image, srcb_labels, tar_image, tar_labels, options, Cs=None):
        running_loss = 0.0
        #src_labels = torch.cat([src_labels[:,0].squeeze(), src_labels[:,1].squeeze()], 0).type(torch.LongTensor).cuda()
        src_labels = torch.cat([srca_labels, srcb_labels]).type(torch.LongTensor).cuda()
        self.model_optim.zero_grad()
        # src image shape batch_size x domain x 3 channels x height x width
        src_out, source_feature = self.generator_model(torch.cat([srca_image, srcb_image]))
        tar_out, target_feature = self.generator_model(tar_image)
        #  Discriminator
        discriminator_x = torch.cat([source_feature, target_feature]).squeeze()
        disc_clf = self.discriminator_model(discriminator_x)
        # Segmentation Losses
        losses = torch.stack([self.generator_criterion(src_out[j*self.batch_size:j*self.batch_size+self.batch_size], src_labels[j*self.batch_size:j*self.batch_size+self.batch_size]) for j in range(self.num_domains)])
        # Wasserstain Losses
        # wass_loss = [self.init_wasserstein.update_wasserstein_multi_source(disc_clf[j*self.batch_size:j*self.batch_size+self.batch_size].squeeze(),disc_clf[self.num_domains*self.batch_size:self.num_domains * self.batch_size+self.batch_size].squeeze()) for j in range(self.num_domains)]
        #domain_losses = torch.stack(wass_loss)
        Xs = [disc_clf[j*self.batch_size:j*self.batch_size+self.batch_size].squeeze() for j in range(self.num_domains)]
        Y = disc_clf[self.num_domains*self.batch_size:self.num_domains * self.batch_size+self.batch_size].squeeze()
        # Different final loss function depending on different training modes.
        # compute gradient penalty
        penalty = self.init_wasserstein.gradient_regularization_multi_source(self.discriminator_model,source_feature.detach(), target_feature.detach(), options['batch_size'], options['num_domains'],
            options['Lf'])
        penalty_0, penalty_1 = self.init_wasserstein.gradient_regularization_multi_source(self.discriminator_model,source_feature.detach(), target_feature.detach(), options['batch_size'], options['num_domains'],
            options['Lf'])
        if options['mode']== "maxmin":
            # src_index x (B x H x W)
            # domain_losses_0 = self.init_wasserstein.update_wasserstein_singlesource( [Xs[0]], Y, torch.Tensor(Cs).cuda())
            # domain_losses_1 = self.init_wasserstein.update_wasserstein_single_source( [Xs[1]], Y, torch.Tensor(Cs).cuda())
            # loss1= torch.mean(losses[0]) - options['mu'] * (domain_losses_0 + options['gamma'] * torch.mean(penalty_0))
            # loss2= torch.mean(losses[1]) - options['mu'] * (domain_losses_1 + options['gamma'] * torch.mean(penalty_1))
            # loss = torch.max(loss1,loss2)
            domain_losses_0 = self.init_wasserstein.update_single_wasserstein(Xs[0], Y).cuda()
            domain_losses_1 = self.init_wasserstein.update_single_wasserstein(Xs[1], Y).cuda()
            penalty_0 = self.init_wasserstein.update_
            domain_losses = self.init_wasserstein.update_wasserstein_multi_source
            torch.max()

        elif options['mode'] == "dynamic":
            # TODO Wasserstein not implemented yet for this
            loss = torch.log(torch.sum(torch.exp(options['gamma'] * (losses + options['mu'] * domain_losses)))) / options['gamma']
        elif options['mode'] == 'default':
            import pdb
            pdb.set_trace()
            domain_losses = self.init_wasserstein.update_wasserstein_multi_source(Xs, Y, torch.Tensor(Cs).cuda())
            loss = torch.mean(losses) - options['mu'] * (torch.mean(domain_losses) + options['gamma'] * penalty)
        else:
            raise ValueError("No support for the training mode on madnNet: {}.".format(options['mode']))
        loss.backward()
        self.model_optim.step()
        ## Additional regularization
        for p in self.discriminator_model.parameters():
               p.data.clamp_(-0.01, 0.01)
        running_loss += loss.detach().cpu().numpy()
        # compute target loss
        target_loss = self.generator_criterion(tar_out, tar_labels).detach().cpu().numpy()
        return running_loss, target_loss
