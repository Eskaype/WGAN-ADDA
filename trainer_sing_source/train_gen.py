import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

#om mypath import Path
from dataset import make_data_loader
from models.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab import *
import torch.nn.functional as F
from models.discriminator import FCDiscriminator
from utils_.loss import SegmentationLosses
# from utils_.calculate_weights import calculate_weigths_labels
from utils_.lr_scheduler import LR_Scheduler
from utils_.saver import Saver
from utils_.summaries import TensorboardSummary
from utils_.metrics import Evaluator

class Adda():

    def __init__(self, args):

           # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.source_loader, self.target_loader, _, self.nclass = make_data_loader(args, **kwargs)

         # Define Target Model
        self.target_model = DeepLab(num_classes=self.nclass,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)

                # Using cuda
        self.best_pred = {'disc': 0.0, 'cup':0.0}

        self.target_model = torch.nn.DataParallel(self.target_model)
        patch_replication_callback(self.target_model)
        self.target_model = self.target_model.cuda()
        model_dict =self.target_model.module.state_dict()
        pretrained_dict = {k:v for k,v in checkpoint['state_dict'].items() if 'last_conv' not in k }
        model_dict.update(pretrained_dict)
        self.target_model.module.load_state_dict(model_dict)
        self.target_model.train()
        self.set_requires_grad('target', True)

        # Define learning rate and optimizer params
        target_params = [{'params': self.target_model.module.get_1x_lr_params(), 'lr': args.lr},
                        {'params': self.target_model.module.get_10x_lr_params(), 'lr': args.lr * 10}]


        target_optim = torch.optim.SGD(target_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        target_optim.zero_grad()

        self.target_criterion =  torch.nn.BCEWithLogitsLoss()
        self.target_optim = target_optim

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.target_loader))
        self.evaluator = Evaluator(3)

    def set_requires_grad(self, mode, requires_grad=False):

        for param in eval('self.'+mode+'_model').parameters():
            param.requires_grad = requires_grad

    def loop_iterable(self, iterable):
        while True:
            yield from iterable
    def trainer(self, num_epochs, iterations):
        best_IoU = 0.68
        attempt = 1
        for epoch in range(1, num_epochs+1):
            total_loss_tgt =0
            total_accuracy = 0
            self.evaluator.reset()
            total_loss = 0
            len_dataloader =len(self.source_loader)
            torch.manual_seed(1 + epoch)
            for step in trange(len_dataloader, leave=True):
                p = float(step + epoch*len_dataloader)/args.epochs/len_dataloader #(1+len)/epochs/
                alpha = 2./(1.+np.exp(-10*p)) - 1
                try:
                    data = next(batch_iterator)
                except StopIteration:
                    batch_iterator = zip(self.loop_iterable(self.source_loader), self.loop_iterable(self.target_loader))
                    data = next(batch_iterator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'glaucoma'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=400,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=400,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='bce',
                        choices=['ce', 'focal', 'bce'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='glaucoma',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default="pretrained/deeplab-resnet.pth.tar",
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default="pretrained/",
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    parser.add_argument('--kdisc', type=int, default=2,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'glaucoma': 100
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)
    args.batch_size = 4
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'glaucoma': 0.07,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    args.lr = 0.0001
    torch.manual_seed(args.seed)
    Adda  = Adda(args)
    #Adda.trainer(400,1)
