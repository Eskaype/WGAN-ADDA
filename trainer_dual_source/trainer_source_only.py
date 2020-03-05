from utils.loss import SegmentationLosses
from models.deeplab import *
from models.unet import *
import torch.nn.functional as F
from utils.lr_scheduler import LR_Scheduler
from models.sync_batchnorm.replicate import patch_replication_callback
import torch
import os
from torch import nn

class multisource_trainer(object):
    def __init__(self, args, nnclass):
        self.generator_model = None
        self.generator_optim = None
        self.generator_criterion = None
        self.batch_size = args.batch_size
        self.nnclass = nnclass
        self.init_generator(args)
        self.init_optimizer(args)

    def init_generator(self, args):

        if args.arch == 'deeplab':
            self.generator_model = DeepLab(num_classes=self.nnclass,
                                backbone='resnet',
                                output_stride=16,
                                sync_bn = None,
                                freeze_bn=False)
        else:
            self.generator_model = UNet(n_channels=3, n_classes=2, bilinear=True)

        self.generator_model = torch.nn.DataParallel(self.generator_model).cuda()
        patch_replication_callback(self.generator_model)
        if args.resume:
            print('load pretrained model')
            model_dict = self.generator_model.module.state_dict()
            checkpoint = torch.load(args.resume)
            pretrained_dict = {k:v for k,v in checkpoint['state_dict'].items() if 'last_conv' not in k and k in model_dict.keys() }
            #pretrained_dict = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()  if 'last_conv' not in k}
            model_dict.update(pretrained_dict)
            self.generator_model.module.load_state_dict(model_dict)
        for param in self.generator_model.parameters():
            param.requires_grad = True

    def init_optimizer(self, args):
        self.generator_criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='bce') #torch.nn.BCELoss(reduce ='mean')
        self.generator_params = [{'params': self.generator_model.module.get_1x_lr_params(), 'lr':args.lr},
                                {'params': self.generator_model.module.get_10x_lr_params(), 'lr': args.lr * 10}]
        self.dda_optim = torch.optim.Adam(self.generator_params)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, lr_step=30, iters_per_epoch=100)

    def update_weights(self, input_, src_labels):
        src_labels = torch.cat([src_labels[:,0].squeeze(), src_labels[:,1].squeeze()], 0).type(torch.LongTensor).cuda()
        input_ = torch.cat([input_[:,0].squeeze(), input_[:,1].squeeze()])
        src_out, source_feature = self.generator_model(input_)
        seg_loss = self.generator_criterion(src_out, src_labels)
        seg_loss.backward()
        self.dda_optim.step()
        self.dda_optim.zero_grad()
        return seg_loss, src_out
