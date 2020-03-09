from utils.loss import SegmentationLosses
from models.deeplab import *
from models.unet import *
import torch.nn.functional as F
from utils.lr_scheduler import LR_Scheduler
from models.discriminator import FCDiscriminator, FCDiscriminator_WGAN
from models.sync_batchnorm.replicate import patch_replication_callback
import torch
import os
from torch import nn
import numpy as np

class multisource_metatrainer(object):
    def __init__(self, args, nnclass, meta_update_lr, meta_update_step, beta, pretrain_mode='meta'):
        self.device = 1
        self.generator_model = None
        self.generator_optim = None
        self.generator_criterion = None
        self.pretrain_mode = pretrain_mode
        self.batch_size = args.batch_size
        self.nnclass = nnclass
        self.init_generator(args)
        self.init_discriminator(args)
        self.init_optimizer(args)
        self.meta_update_lr = meta_update_lr
        self.meta_update_step = meta_update_step
        self.beta = beta

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
    def update_weights(self, srca, srca_labels, src_b , srcb_labels, target_img, target_label):
        #self.pretrain_mode = 'meta'
        src_labels = torch.cat([srca_labels.squeeze(), srcb_labels.squeeze()], 0).type(torch.LongTensor).cuda()
        src_image = torch.cat([srca.squeeze(), src_b.squeeze()])
        if self.pretrain_mode =='meta':
            seg_loss = self.meta_mldg(src_image, src_labels, self.batch_size)
        else:
            print('a default training is enabled')
            src_out, source_feature = self.generator_model(src_image)
            seg_loss = self.generator_criterion(src_out, src_labels)
        self.model_optim.zero_grad()
        seg_loss.backward()
        self.model_optim.step()
        target_logit,_ = self.generator_model(target_img.cuda())
        tgt_loss = self.generator_criterion(target_logit, target_label)
        tgt_loss = tgt_loss.detach()
        seg_loss = seg_loss.detach()
        return seg_loss, tgt_loss


    def meta_mldg(self, src_image, src_labels, batch_size):
        batch_size = 4
        num_src = 2
        S = np.random.choice(num_src)
        V = abs(S-1)
        source_out, _ = self.generator_model(src_image[S*batch_size:(S+1)*batch_size].squeeze())
        losses = self.generator_criterion(source_out, src_labels[S*batch_size:(S+1)*batch_size])
        for k in range(1, self.meta_update_step):
            source_out, _ = self.generator_model(src_image[S*batch_size:(S+1)*batch_size].squeeze())
            loss = self.generator_criterion(source_out, src_labels[S*batch_size:(S+1)*batch_size])
            grad = torch.autograd.grad(loss, self.generator_model.parameters())
            fast_weights = list(map(lambda p: p[1] - self.meta_update_lr * p[0], zip(grad, self.generator_model.parameters())))
            # compute the test loss on the fast weights
            Grad_test = self.generator_model(src_image[V*batch_size:(V+1)*batch_size], fast_weights, bn_training=True)
            # compute the gradient on generator_model
            losses += self.beta*Grad_test
        return losses




    # def meta(self, src_out, src_labels):
    #     """
    #     :param x_spt:   [b, setsz, c_, h, w]
    #     :param y_spt:   [b, setsz]

    #     :return:
    #     """
    #     x_spt =
    #     y_spt =


    #     num_src = 2
    #     for i in range(num_src):
    #         # 1. run the i-th task and compute loss for k=0
    #         logits = self.generator_model(x_spt[i], vars=None, bn_training=True)
    #         loss = self.generator_criterion(logits, y_spt[i])
    #         grad = torch.autograd.grad(loss, self.generator_model.pameters())
    #         fast_weights = list(map(lambda p: p[1] - self.meta_update_lr * p[0], zip(grad, self.generator_model.parameters())))
    #         # # this is the loss and accuracy before first update
    #         # with torch.no_grad():
    #         #     # [setsz, nway]
    #         #     logits_q = self.generator_model(x_qry[i], self.generator_model.parameters(), bn_training=True)
    #         #     loss_q = self.generator_criterion(logits_q, y_qry[i])
    #         # # this is the loss and accuracy after the first update
    #         # with torch.no_grad():
    #         #     # [setsz, nway]
    #         #     logits_q = self.generator_model(x_qry[i], fast_weights, bn_training=True)
    #         #     loss_q = self.generator_criterion(logits_q, y_qry[i])
    #         losses_q = 0
    #         for k in range(1, self.meta_update_step):
    #             # 1. run the i-th task and compute loss for k=1~K-1
    #             logits = self.generator_model(x_spt[i], fast_weights, bn_training=True)
    #             loss = self.generator_criterion(logits, y_spt[i])
    #             # 2. compute grad on theta_pi
    #             grad = torch.autograd.grad(loss, fast_weights)
    #             # 3. theta_pi = theta_pi - train_lr * grad
    #             fast_weights = list(map(lambda p: p[1] - self.meta_update_lr * p[0], zip(grad, fast_weights)))
    #             logits_q = self.generator_model(x_spt[i], fast_weights, bn_training=True)
    #             # loss_q will be overwritten and just keep the loss_q on last update step.
    #             losses_q += self.generator_criterion(logits_q, y_spt[i])
    #     return losses_q
