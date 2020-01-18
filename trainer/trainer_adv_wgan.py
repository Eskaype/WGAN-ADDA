from utils_.wasserstein import Wassterstein
from utils_.loss import SegmentationLosses
from models.deeplab import *
import torch.nn.functional as F
from models.discriminator import FCDiscriminator, FCDiscriminator_WGAN
from utils_.lr_scheduler import LR_Scheduler
from models.sync_batchnorm.replicate import patch_replication_callback
from dataset.adversarial_augmentation import FastGradientSignUntargeted
import torch
import os
from torch import nn

class wgan_trainer(object):
    def __init__(self, args, nnclass):
        self.wsn = Wassterstein()
        self.target_model = None
        self.target_optim = None
        self.target_criterion = None
        self.batch_size = args.batch_size
        self.nnclass = nnclass
        self.init_target(args)
        self.init_discriminator(args)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, lr_step=30, iters_per_epoch=100)
        self.disc_params=[{'params': self.disc_model.parameters(), 'lr': args.lr*5}]
        self.dda_optim = torch.optim.Adam(self.train_params)
        #self.dda_optim = torch.optim.SGD(self.train_params, momentum=args.momentum,
        #                            weight_decay=args.weight_decay, nesterov=args.nesterov)
        #self.discriminator_optim = torch.optim.SGD(self.disc_params, momentum=args.momentum,
        #                            weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.discriminator_optim = torch.optim.Adam(self.disc_params)
        # self.adv_aug = FastGradientSignUntargeted( self.target_model,
        #                                             0.0157,
        #                                             0.00784,
        #                                             min_val=-2,
        #                                             max_val=2,
        #                                             max_iters=5,
        #                                             _type='linf')

    def init_target(self, args):

        self.target_model = DeepLab(num_classes=self.nnclass,
                            backbone='resnet',
                            output_stride=16,
                            sync_bn = None,
                            freeze_bn=False)
        self.train_params = [{'params': self.target_model.get_1x_lr_params(), 'lr':args.lr},
                    {'params': self.target_model.get_10x_lr_params(), 'lr': args.lr * 10}]
        self.target_model = torch.nn.DataParallel(self.target_model)
        self.target_criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='bce') #torch.nn.BCELoss(reduce ='mean')
        patch_replication_callback(self.target_model)
        model_dict = self.target_model.module.state_dict()
        checkpoint = torch.load(args.resume)
        #pretrained_dict = {k:v for k,v in checkpoint['state_dict'].items() if 'last_conv' not in k }
        pretrained_dict = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items() }
        model_dict.update(pretrained_dict)
        self.target_model.module.load_state_dict(model_dict)
        self.target_model = self.target_model.cuda()
        return

    def init_discriminator(self, args):
        # init D
        self.disc_model = FCDiscriminator_WGAN(num_classes=2).cuda()
        self.interp = nn.Upsample(size=400, mode='bilinear')
        self.disc_criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        return

    def update_weights(self, input_, src_labels, target, tgt_labels, lamda_g, gamma, trainmodel):

        wasserstein_distance = 0
        self.dda_optim.zero_grad()
        self.discriminator_optim.zero_grad()
        if trainmodel == 'train_gen':
            for param in self.target_model.parameters():
                param.requires_grad = True
            for param in self.disc_model.parameters():
                param.requires_grad = False
            self.disc_model.eval()
            self.target_model.train()
        else:
            for param in self.target_model.parameters():
                param.requires_grad = False
            for param in self.disc_model.parameters():
                param.requires_grad = True
            self.disc_model.train()
            self.target_model.eval()

        src_out, source_feature = self.target_model(input_)
        seg_loss = self.target_criterion(src_out, src_labels)
        targ_out, target_feature = self.target_model(target)

        # discriminator
        discriminator_x = torch.cat([source_feature, target_feature]).squeeze()

        #discriminator_adv_logit = torch.cat([torch.zeros(source_feature.shape),
        #                                torch.ones(target_feature.shape)])
        #discriminator_real_logit = torch.cat([torch.ones(source_feature.shape),
        #                                 torch.zeros(target_feature.shape)])
        #disc_out, disc_clf_src = self.disc_model(source_feature)
        #disc_out, disc_clf_targ = self.disc_model(target_feature)
        _ , disc_clf = self.disc_model(discriminator_x)
        import pdb
        #disc_loss = self.disc_criterion(disc_out, discriminator_real_logit.cuda())
        wasdist_src = 0
        wasdist_targ = 0
        regu_val_disc= 0
        for i,lab in enumerate(['disc', 'cup']):
            D_src_out_classes = disc_clf[0: self.batch_size].squeeze()#*src_labels[:, i, :, :] # shape: BXWXH & BXWXHX2
            D_tar_out_classes = disc_clf[self.batch_size:].squeeze()#*targ_out[:, i, :, :].detach() # shape: BXWXH & BXWXHX2
            dist_s, dist_targ= self.wsn.update_single_wasserstein(D_src_out_classes, D_tar_out_classes, src_labels[:, i, :, :], targ_out[:, i, :, :].detach())
            if lab == 'cup':
                wasdist_src+=dist_s
                wasdist_targ+=dist_targ
            else:
                wasdist_src+=dist_s
                wasdist_targ+=dist_targ

        if trainmodel == 'train_gen':
            loss_seg = seg_loss  - lamda_g * (wasdist_src - wasdist_targ)
            loss_seg.backward()
            self.dda_optim.step()
            tgt_loss = self.target_criterion(targ_out, tgt_labels)
            #rint(seg_loss)
            return seg_loss.data.cpu().numpy(), tgt_loss.data.cpu().numpy()
        else:
            gp = 0
            # for bats in range(4):
            #     gp += self.wsn.gradient_penalty(self.disc_model, source_feature.detach()[i,].unsqueeze(0) , target_feature.detach()[i,].unsqueeze(0))
            regu_val_disc, regu_val_cup  = self.wsn.gradient_regularization(self.disc_model, source_feature.detach() , target_feature.detach())
            disc_loss = wasdist_src - wasdist_targ + gamma*regu_val_disc + gamma*regu_val_cup
            disc_loss.backward()
            self.discriminator_optim.step()
            for p in self.disc_model.parameters():
               p.data.clamp_(-0.01, 0.01)
            tgt_loss = self.target_criterion(targ_out, tgt_labels)
            #rint(seg_loss)
            return seg_loss.data.cpu().numpy(), tgt_loss.data.cpu().numpy(), regu_val_disc.detach().cpu().numpy()
