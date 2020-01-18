from utils_.wasserstein import Wassterstein
from utils_.loss import SegmentationLosses
from models.deeplab import *
import torch.nn.functional as F
from models.discriminator import FCDiscriminator
from utils_.lr_scheduler import LR_Scheduler
from models.sync_batchnorm.replicate import patch_replication_callback
from dataset.adversarial_augmentation import FastGradientSignUntargeted
import torch
import os
from torch import nn

class adda_trainer(object):
    def __init__(self, args, nnclass):
        self.target_model = None
        self.target_optim = None
        self.target_criterion = None
        self.batch_size = args.batch_size
        self.nnclass = nnclass
        self.init_target(args)
        self.init_discriminator(args)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, lr_step=40, iters_per_epoch=100)
        self.disc_params=[{'params': self.disc_model.parameters(), 'lr': args.lr*5 }]
        self.dda_optim = torch.optim.Adam(self.train_params)
        self.discriminator_optim = torch.optim.Adam(self.disc_params)
        #self.dda_optim = torch.optim.SGD(self.train_params, momentum=args.momentum,
        #                            weight_decay=args.weight_decay, nesterov=args.nesterov)
        #self.discriminator_optim = torch.optim.SGD(self.disc_params, momentum=args.momentum,
        #                            weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.adv_aug = FastGradientSignUntargeted( self.target_model,
                                         0.0157,
                                         0.00784,
                                         min_val=0,
                                         max_val=1,
                                         max_iters=2 ,
                                         _type='linf')

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
        pretrained_dict = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items() }
        #pretrained_dict = {k:v for k,v in checkpoint['state_dict'].items() if 'last_conv' not in k }
        model_dict.update(pretrained_dict)
        self.target_model.module.load_state_dict(model_dict)
        self.target_model = self.target_model.cuda()
        return

    def init_discriminator(self, args):
        # init D
        self.disc_model = FCDiscriminator(num_classes=2).cuda()
        self.interp = nn.Upsample(size=400, mode='bilinear')
        self.disc_criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        return

    def update_weights(self, input_, src_labels, target, tgt_labels, lamda_g, trainmodel):

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
        #tot_input = torch.cat([input_, target])
        #import pdb
        #pdb.set_trace()
        src_out, source_feature = self.target_model(input_)
        seg_loss = self.target_criterion(src_out, src_labels)
        #print(target.shape)
        targ_out, target_feature = self.target_model(target)

        # discriminator
        discriminator_x = torch.cat([source_feature, target_feature]).squeeze()
        discriminator_adv_logit = torch.cat([torch.zeros(source_feature.shape),
                                         torch.ones(target_feature.shape)])
        discriminator_real_logit = torch.cat([torch.ones(source_feature.shape),
                                         torch.zeros(target_feature.shape)])
        disc_out = self.disc_model(discriminator_x)
        #print(source_feature.shape, input_.shape,discriminator_adv_logit.shape, disc_out.shape)
        adv_loss= self.target_criterion(disc_out, discriminator_adv_logit[:,0,:,:].cuda())
        adv_loss+= self.target_criterion(disc_out, discriminator_adv_logit[:,1,:,:].cuda())
        disc_loss= self.disc_criterion(disc_out, discriminator_real_logit[:,0,:,:].cuda())
        disc_loss+= self.disc_criterion(disc_out, discriminator_real_logit[:,1,:,:].cuda())
        if trainmodel == 'train_gen':
            loss_seg = seg_loss + lamda_g * adv_loss
            loss_seg.backward()
            self.dda_optim.step()
        else:
            disc_loss.backward()
            self.discriminator_optim.step()
        tgt_loss = self.target_criterion(targ_out, tgt_labels)
        return seg_loss.data.cpu().numpy(), tgt_loss.data.cpu().numpy()
