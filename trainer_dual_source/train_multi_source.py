import argparse
import os
import numpy as np
import random
from tqdm import tqdm, trange
from PIL import Image
import matplotlib.pyplot as plt
from datasets import make_data_loader
from utils.metrics import compute_iou
from utils.test_sanity import sanity_check,sanity_check_2
from trainer.trainer_multisource import multisource_trainer

import torch
class multi_source:
    def __init__(self, args):
        kwargs = {'num_workers': 4, 'pin_memory': True}
        self.num_class = 2
        self.source_loader, self.target_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        self.tbar = tqdm(self.test_loader, desc='\r')
        self.best_IoU = {'disc': 0.77, 'cup': 0.65}
        self.attempt = 9.5
        self.multisource_trainer = multisource_trainer(args, self.num_class)
        self.trainer_multisource(args)

    def loop_iterable(self, iterable):
        while True:
            yield from iterable

    def save_model(self, epoch, IoU):
        print('---- MODEL SAVE ---')
        torch.save({'epoch': epoch + 1, 'state_dict': self.multisource_trainer.target_model.state_dict(), 'best_auc': str(self.best_IoU['cup']),
                    'optimizer' : self.multisource_trainer.dda_optim.state_dict()}, 'best_origa/m-adda_wgan_clip_0.03' + "v_" + str(self.attempt) + '.pth.tar')
        return

    def trainer_multisource(self, args):
        print("trainer initialized training started")
        for epoch in range(args.epochs):
            self.validation(epoch)
            self.multisource_trainer.generator_model.train()
            total_loss = 0
            len_dataloader = len(self.source_loader)
            torch.manual_seed(1 + epoch)
            for step, source_data in enumerate(self.source_loader):
                image = source_data[0].cuda()
                mask  = source_data[1].cuda()
                batch_loss,_ = self.multisource_trainer.update_weights(image, mask)
                total_loss+=batch_loss.detach().cpu().numpy()
                if step %50 ==0:
                    print('batch wise loss {} at batch {}'.format(total_loss/(step+1), step+1))
            print("total epoch loss {}".format(total_loss/(step+1)))
        return

    def validation(self, epoch):
        self.multisource_trainer.generator_model.eval()
        test_loss = 0.0
        predict_disc =None
        target_disc = None
        predict_cup = None
        target_cup = None
        for i, data in enumerate(self.tbar):
            image, target = data[0], data[1]
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output,_ = self.multisource_trainer.generator_model(image)
            test_loss = self.multisource_trainer.generator_criterion(output, target)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            if i==0:
                target_disc = target[:,0,].squeeze()
                target_cup = target[:,1,].squeeze()
                predict_disc = pred[:,0,].squeeze()
                predict_cup= pred[:,1,].squeeze()
            else:
                target_disc = np.vstack([target_disc, target[:,0,].squeeze()])
                target_cup = np.vstack([target_cup, target[:,1,].squeeze()])
                predict_disc = np.vstack([predict_disc, pred[:,0,].squeeze()])
                predict_cup = np.vstack([predict_cup, pred[:,1,].squeeze()])
        #evaluator.Plot_Loss(1)
        print('Validation on total  set of size {}'.format(len(target_disc)))
        #print('[Epoch: %d, numImages: %5d]' % (epoch, i * args.batch_size + image.data.shape[0]))
        #if epoch == 1:
        #   sanity_check_2(image.detach().cpu().numpy(), target_disc, predict_disc)
        iou_cup = compute_iou(target_cup, predict_cup)
        iou_disc = compute_iou(target_disc, predict_disc)
        print("for Epoch {} iou disc:{} and iou_cup:{}".format(epoch , iou_disc, iou_cup))
        if iou_cup > self.best_IoU['cup']:
            self.best_IoU['cup'] = iou_cup
            self.best_IoU['disc'] = iou_disc
            print("best iou is {} on epoch {}".format(iou_cup, epoch))
            if args.save_model == True:
                save_model(epoch)
                print("best model saved at {}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=list, default=['refuge', 'origa'],
                        choices=['refuge', 'origa', 'dristhi'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--source1_dataset', type=str, default='/storage/shreya/datasets/glaucoma/split_ORIGA/',
                        help='dataset name (default: pascal)')
    parser.add_argument('--source2_dataset', type=str, default='/storage/shreya/datasets/glaucoma/split_refuge/',
                        help='dataset name (default: pascal)')
    parser.add_argument('--target_dataset', type=str, default='/storage/shreya/datasets/glaucoma/split_drishti/',
                        help='dataset name (default: pascal)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
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

    parser.add_argument('--loss-type', type=str, default='bce',
                        choices=['ce', 'focal', 'bce'],
                        help='loss func type (default: ce)')
    parser.add_argument('--lr_critic', type=int, default=1e-4,
                        help='skip validation during training')
    parser.add_argument('--gamma', type=int, default=10,
                        help='skip validation during training')
    parser.add_argument('--lambda_g', type=int, default=1,
                        help='skip validation during training')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: auto)')

    parser.add_argument('--k_disc', type=int, default=1,
                        help='skip validation during training')
    parser.add_argument('--k_src', type=int, default=1,
                        help='skip validation during training')
    parser.add_argument('--k_targ', type=int, default=1,
                        help='skip validation during training')
    # checking point
    parser.add_argument('--resume', type=str, default= 'pretrained/deeplab-resnet.pth.tar',#"m-adda_wganv_9.1.pth.tar", #"run/glaucoma/best_experiment_2.pth.tar",
                        help='put the path to resuming file if needed')
    parser.add_argument('--save_model', type=bool, default= 'False',#"m-adda_wganv_9.1.pth.tar", #"run/glaucoma/best_experiment_2.pth.tar",
                        help='put the path to resuming file if needed')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    args.batch_size = 4
    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'glaucoma': 0.007,
        }
    args.lr = 1e-4 # 5e-5 best model
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    multi_source(args)

main()
