import argparse
import os
import numpy as np
import random
from tqdm import tqdm, trange
from PIL import Image
import matplotlib.pyplot as plt

#from mypath import Path
from dataset import make_data_loader
import torch.nn.functional as F
from utils_.saver import Saver
from utils_.summaries import TensorboardSummary
from utils_.metrics import Evaluator
from trainer_nanqing import *
from utils_.visualize import visualize_plot

class adda:
    def __init__(self, args):
        kwargs = {'num_workers': 4, 'pin_memory': True}
        self.source_loader, self.target_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        self.tbar = tqdm(self.test_loader, desc='\r')
        self.trainer = adda_trainer(args, 2)
        self.evaluator = Evaluator(2)
        self.best_IoU = {'disc': 0.77, 'cup': 0.60}
        self.attempt = 8.2
        #logger([self.lr, ])
        self.validation(args, self.trainer.target_model, self.tbar )
        self.trainer_dda(args)

    def loop_iterable(self, iterable):
        while True:
            yield from iterable


    def save_model(self, epoch):
        print('Validation:')
        Acc = self.evaluator.Pixel_Accuracy([self.evaluator.confusion_matrix_disc, self.evaluator.confusion_matrix_cup])
        Acc_class = self.evaluator.Pixel_Accuracy_Class([self.evaluator.confusion_matrix_disc, self.evaluator.confusion_matrix_cup])
        mIoU = self.evaluator.Mean_Intersection_over_Union([self.evaluator.confusion_matrix_disc, self.evaluator.confusion_matrix_cup])
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union([self.evaluator.confusion_matrix_disc, self.evaluator.confusion_matrix_cup])
        print("epoch:{}, Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(epoch, Acc, Acc_class, mIoU, FWIoU))
        if ( mIoU['cup'] > self.best_IoU['cup']):
            #model save
            self.best_IoU = mIoU
            print('---- MODEL SAVE ---')
            torch.save({'epoch': epoch + 1, 'state_dict': self.trainer.target_model.state_dict(), 'best_auc': str(mIoU['cup']),
                        'optimizer' : self.trainer.dda_optim.state_dict()}, 'm-adda' + "v_" +str(self.attempt) + '.pth.tar')
        return mIoU

    def trainer_dda(self, args):
        self.trainer.target_model.train()
        self.trainer.disc_model.train()
        self.evaluator.reset()
        max_epochs = args.epochs
        for epoch in range(1, max_epochs+1):
            self.trainer.target_model.train()
            batch_iterator = zip(self.loop_iterable(self.source_loader), self.loop_iterable(self.target_loader))
            total_loss = 0
            total_loss_tgt = 0
            loss_critic = 0
            loss_tgt = 0
            total_accuracy = 0
            len_dataloader = max(len(self.source_loader), len(self.target_loader))
            torch.manual_seed(1 + epoch)
            for step in trange(len_dataloader, leave=True):
                p = float(step + epoch*len_dataloader)/args.epochs/len_dataloader #(1+len)/epochs/
                alpha = 2./(1.+np.exp(-10*p)) - 1
                try:
                    data = next(batch_iterator)
                except StopIteration:
                    batch_iterator = zip(self.loop_iterable(self.source_loader), self.loop_iterable(self.target_loader))
                    data = next(batch_iterator)
                if epoch < 0:
                    source_x, src_labels = data[0][0].cuda(), data[0][1].cuda()
                    target_x, target_lab = data[1][0].cuda(),  data[1][1].cuda()
                    dda_loss, tgt_loss = self.trainer.update_weights(source_x, src_labels, target_x, target_lab, 0, 'train_gen')
                    continue
                for i in range(args.k_disc):
                    source_x, src_labels = data[0][0].cuda(), data[0][1].cuda()
                    target_x, target_lab = data[1][0].cuda(),  data[1][1].cuda()
                    #target_x = self.trainer.adv_aug.perturb(target_x, target_lab, self.trainer.target_criterion, random_start=False )
                    dda_loss, tgt_loss = self.trainer.update_weights(source_x, src_labels, target_x, target_lab, 0.1,'train_disc')

                for i in range(args.k_src):
                    source_x, src_labels = data[0][0].cuda(), data[0][1].cuda()
                    target_x, target_lab = data[1][0].cuda(),  data[1][1].cuda()
                    #target_x = self.trainer.adv_aug.perturb(target_x, target_lab, self.trainer.target_criterion, random_start=False )
                    dda_loss, tgt_loss = self.trainer.update_weights(source_x, src_labels, target_x, target_lab, 0.1,'train_gen')
                total_loss+=dda_loss
                total_loss_tgt +=tgt_loss
                #if(step%100 == 0):
                #    print("Target_loss:{}, disc_loss:{}".format(total_loss_tgt/(step+1), total_loss/(step+1)))
                self.trainer.scheduler(self.trainer.dda_optim, step, epoch, self.best_IoU['cup'])
            self.trainer.target_model.eval()
            self.trainer.disc_model.eval()
            for st, data in enumerate(self.tbar):
                image, target = data[0], data[1]
                image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    output,_ = self.trainer.target_model(image)
                test_loss = self.trainer.target_criterion(output, target)
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                # Add batch sample into evaluator
                self.evaluator.add_batch(target, pred)
                self.evaluator.add_test_loss(test_loss/(st+1))
            mIoU = self.evaluator.Mean_Intersection_over_Union([self.evaluator.confusion_matrix_disc, self.evaluator.confusion_matrix_cup])
            print("mIoU:{}".format(mIoU))
            total_accuracy =0
            if ((epoch + 1) % 1== 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                            "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                            .format(epoch + 1,
                                    max_epochs,
                                    epoch + 1,
                                    len(self.source_loader),
                                    total_loss/((step+1)),
                                    total_loss_tgt/(step+1),total_accuracy))
            mIoU = self.save_model(epoch)


    def validation(self, args, model, tbar):
        best_pred = {'cup':0, 'disc':0}
        model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        for i, data in enumerate(tbar):
            image, target = data[0], data[1]
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output,_ = model(image)
            test_loss = self.trainer.target_criterion(output, target)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            self.evaluator.add_test_loss(test_loss/(i+1))

        mIoU = self.evaluator.Mean_Intersection_over_Union([self.evaluator.confusion_matrix_disc, self.evaluator.confusion_matrix_cup])
        #evaluator.Plot_Loss(1)
        print('Validation:')
        #print('[Epoch: %d, numImages: %5d]' % (epoch, i * args.batch_size + image.data.shape[0]))
        print("mIoU:{}".format(mIoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU
        if new_pred['cup'] > best_pred['cup']:
            is_best = True
            best_pred = new_pred
def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='glaucoma',
                        choices=['pascal', 'coco', 'cityscapes', 'glaucoma'],
                        help='dataset name (default: pascal)')

    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                training (default: auto)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
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
    #parser.add_argument('--resume', type=str, default= "pretrained/deeplab-resnet.pth.tar", help='put the path to resuming file if needed')
    parser.add_argument('--resume', type=str, default= "m-adda_wganv_9.1.pth.tar", #"run/glaucoma/best_experiment_2.pth.tar",
                        help='put the path to resuming file if needed')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    args.batch_size = 2
    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'glaucoma': 0.007,
        }
    args.lr = 3e-6
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    adda(args)

main()
