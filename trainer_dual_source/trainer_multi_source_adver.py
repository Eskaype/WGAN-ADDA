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

class adversarial_madan_trainer(object):
    def __init__(self, args, nnclass, ndomains, loss_type):
        self.loss_type = loss_type
        if self.loss_type == 'wasserstein':
            self.wsn = Wasserstein()
        self.device = 1
        self.generator_model = None
        self.generator_optim = None
        self.generator_criterion = None
        self.batch_size = args.batch_size
        self.nnclass = nnclass
        self.num_domains = ndomains
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
        self.discriminator_model = FCDiscriminator_WGAN(num_classes=2).cuda()
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

    # DANN Wassersteian
    def update_weights_wasserstein(self, src_image, src_labels, targ_image, targ_labels, options):
        running_loss = 0.0
        src_labels = torch.cat([src_labels[:,0].squeeze(), src_labels[:,1].squeeze()], 0).type(torch.LongTensor).cuda()
        self.model_optim.zero_grad()
        # src image shape batch_size x domain x 3 channels x height x width
        src_out, source_feature = self.generator_model(torch.cat([src_image[:,0].squeeze(), src_image[:,1].squeeze()]))
        targ_out, target_feature = self.generator_model(targ_image)
        #  Discriminator
        discriminator_x = torch.cat([source_feature, target_feature]).squeeze()
        _, disc_clf = self.discriminator_model(discriminator_x)
        # Losses
        losses = torch.stack([self.generator_criterion(src_out[j*self.batch_size:j+self.batch_size], src_labels[j*self.batch_size:j+self.batch_size]) for j in range(self.num_domains)])
        # Wassterstein distance
        domain_losses = torch.stack([self.wsn.update_single_wasserstein(disc_clf[j*self.batch_size:j+self.batch_size].squeeze(),
                                     disc_clf[2*self.batch_size+j*self.batch_size: 2*self.batch_size+(j+1)*self.batch_size].squeeze())
                                     for j in range(self.num_domains)])
        # Different final loss function depending on different training modes.
        if options['mode']== "maxmin":
            loss = torch.max(losses) + options['mu'] * torch.min(domain_losses) + gradient_penalty
        elif options['mode'] == "dynamic":
            loss = torch.log(torch.sum(torch.exp(options['gamma'] * (losses + options['mu'] * domain_losses)))) / options['gamma']
        else:
            raise ValueError("No support for the training mode on madnNet: {}.".format(options['mode']))
        loss.backward()
        self.model_optim.step()
        running_loss += loss.detach().cpu().numpy()
        # compute target loss
        target_loss = self.generator_criterion(targ_out, targ_labels).detach().cpu().numpy()
        return running_loss, target_loss

    ## ADDA
    def update_weights_adversarial(self, src_image, src_labels, targ_image, targ_labels, options):
        # Adversarial loss
        self.model_optim.zero_grad()
        if options['trainmodel'] == 'train_gen':
            for param in self.generator_model.parameters():
                param.requires_grad = True
            for param in self.discriminator_model.parameters():
                param.requires_grad = False
            self.discriminator_model.eval()
            self.generator_model.train()
        else:
            for param in self.generator_model.parameters():
                param.requires_grad = False
            for param in self.discriminator_model.parameters():
                param.requires_grad = True
            self.discriminator_model.train()
            self.generator_model.eval()
        src_labels = torch.cat([src_labels[:,0].squeeze(), src_labels[:,1].squeeze()], 0).type(torch.LongTensor).cuda()
        # src image shape batch_size x domain x 3 channels x height x width
        src_out, source_feature = self.generator_model(torch.cat([src_image[:,0].squeeze(), src_image[:,1].squeeze()]))
        targ_out, target_feature = self.generator_model(targ_image)
        #  Discriminator
        discriminator_x = torch.cat([source_feature, target_feature]).squeeze()
        disc_clf, _ = self.discriminator_model(discriminator_x)
        # Losses
        losses = torch.stack([self.generator_criterion(src_out[j*self.batch_size:j+self.batch_size], src_labels[j*self.batch_size:j+self.batch_size]) for j in range(self.num_domains)])
        # adversaril labels
        advslabels = torch.zeros(self.batch_size, disc_clf.shape[2], disc_clf.shape[3], requires_grad=False).type(torch.LongTensor).cuda()
        advtlabels = torch.ones(self.batch_size*2, disc_clf.shape[2], disc_clf.shape[3], requires_grad=False).type(torch.LongTensor).cuda()
        # real labels
        realslabels = torch.ones(self.batch_size, disc_clf.shape[2], disc_clf.shape[3], requires_grad=False).type(torch.LongTensor).cuda()
        realtlabels = torch.zeros(self.batch_size*2, disc_clf.shape[2], disc_clf.shape[3], requires_grad=False).type(torch.LongTensor).cuda()
        domain_losses = 0
        if options['trainmodel'] == 'train_gen':
            domain_losses = torch.stack([self.generator_criterion(disc_clf[j*self.batch_size:j+self.batch_size].squeeze(), advslabels) for j in range(self.num_domains)])
            domain_losses = torch.cat([domain_losses, self.generator_criterion(disc_clf[2*self.batch_size:2*self.batch_size+2*self.batch_size].squeeze(), advtlabels).view(-1)])
            gen_loss = torch.max(losses) + options['mu'] * torch.max(domain_losses)
            gen_loss.backward()
            self.model_optim.step()
        else:
            domain_losses = torch.stack([self.generator_criterion(disc_clf[j*self.batch_size:j+self.batch_size].squeeze(), realslabels) for j in range(self.num_domains)])
            domain_losses = torch.cat([domain_losses, self.generator_criterion(disc_clf[2*self.batch_size:2*self.batch_size+2*self.batch_size].squeeze(), realtlabels).view(-1)])
            disc_loss = options['gamma'] * torch.max(domain_losses)
            disc_loss.backward()
            self.model_optim.step()
        running_loss+= disc_loss.detach().cpu().numpy()
        # compute target loss
        target_loss = self.generator_criterion(targ_out, targ_labels).detach().cpu().numpy()
        return running_loss, target_loss