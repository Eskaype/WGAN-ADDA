import argparse
import pdb
import os
import numpy as np
import random
from tqdm import tqdm, trange
from PIL import Image
import matplotlib.pyplot as plt
from datasets import make_data_loader
from utils.metrics import compute_iou, compute_cdr
from utils.test_sanity import sanity_check, sanity_check_2, check_preprocess_sanity
from trainer_dual_source.trainer_pretrain_meta import multisource_metatrainer
import torch
import time

class multi_source:
    def __init__(self, args):
        kwargs = {'num_workers': 4, 'pin_memory': True}
        self.device = 1
        self.num_class = 2
        self.num_domains = 2
        self.srca_train_loader, self.srca__val_loader, self.srca__test_loader, self.nclass = make_data_loader(0, args.source1_dataset, args, **kwargs)
        self.srcb_train_loader,self.srcb_val_loader, self.srcb_test_loader, self.nclass = make_data_loader(1, args.source2_dataset, args, **kwargs)
        self.tgt_train_loader, self.tgt_val_loader, self.tgt_test_loader, self.nclass = make_data_loader(2, args.target_dataset, args, **kwargs)
        self.tbar = tqdm(self.tgt_val_loader, desc='\r')
        self.best_metrics = {'disc': 0.77, 'cup': 0.65,  'delta_cdr': 1.0}
        self.hyper_dict = args.hyparams_dict
        meta_update_lr = 1e-4
        meta_update_step = 1
        beta = 0.001
        self.mwdan_trainer = multisource_metatrainer(args, self.num_class, meta_update_lr, meta_update_step, beta)

    def loop_iterable(self, iterable):
        while True:
            yield from iterable

    def save_model(self, epoch, iou_disc, iou_cup, delta_cdr, timestampLaunch, hyper_dict, save_dir='multi_wgan_clip_0.03'):
        print('---- MODEL SAVE ---')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        torch.save({'epoch': epoch + 1, 'state_dict': self.mwdan_trainer.generator_model.state_dict(), 'iou_disc': str(iou_disc), 'iou_cup': str(iou_cup), 'delta_cdr': str(delta_cdr), 'optimizer' : self.mwdan_trainer.model_optim.state_dict(), 'hyperparam_dict': hyper_dict}, '{}/{}_{}_{}_{}.pth.tar'.format(save_dir, timestampLaunch, iou_disc, iou_cup, delta_cdr, epoch))
        return

    def train_meta_source(self, args):
        print("trainer initialized training started")
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampLaunch = timestampDate + '-' + timestampTime
        for epoch in range(args.epochs):
            self.validation(args, timestampLaunch, epoch)
            self.mwdan_trainer.generator_model.train()
            total_loss = 0
            batch_iterator_sourcea = enumerate(self.srca_train_loader)
            batch_iterator_sourceb = enumerate(self.srcb_train_loader)
            batch_iterator_target = enumerate(self.tgt_train_loader)
            torch.manual_seed(1 + epoch)
            len_dataloader = max(len(self.srca_train_loader), len(self.srcb_train_loader))
            for step in trange(len_dataloader, leave=True):
                data_srca = next(batch_iterator_sourcea)
                try:
                    data_srcb = next(batch_iterator_sourceb)
                except StopIteration:
                    batch_iterator_sourceb = enumerate(self.srcb_train_loader)
                    data_srcb = next(batch_iterator_sourceb)
                try:
                    data_tar = next(batch_iterator_target)
                except StopIteration:
                    batch_iterator_target = enumerate(self.tgt_train_loader)
                    data_tar = next(batch_iterator_target)
                sourcea_x, srca_labels = data_srca[1][0].cuda(), data_srca[1][1].cuda()
                sourceb_x, srcb_labels = data_srcb[1][0].cuda(), data_srcb[1][1].cuda()
                target_x, tar_labels = data_tar[1][0].cuda(),  data_tar[1][1].cuda()
                source_loss, target_loss = self.mwdan_trainer.update_weights(sourcea_x, srca_labels, sourceb_x, srcb_labels, target_x, tar_labels) #0.2, 0.01
                total_loss += target_loss
                if step % 50 ==0:
                    print('batch wise loss {} at batch {}'.format(total_loss/(step+1), step+1))
            print("total epoch loss {}".format(total_loss/(step+1)))
            self.validation(args, timestampLaunch, epoch, Cs)
        return

    def validation(self, args, timestampLaunch, epoch):
        self.mwdan_trainer.generator_model.eval()
        test_loss = 0.
        for i, data in enumerate(self.tbar):
            image, target = data[0], data[1]
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output, _ = self.mwdan_trainer.generator_model(image)
            test_loss += self.mwdan_trainer.generator_criterion(output, target)
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
        iou_cup, dic_cup = compute_iou(target_cup, predict_cup)
        iou_disc, dic_disc = compute_iou(target_disc, predict_disc)
        try:
            delta_cdr = compute_cdr(predict_cup, predict_disc, target_cup, target_disc)
        except Exception:
            delta_cdr = 10
        print("for Epoch {} iou disc:{} and iou_cup:{} and delta_cdr:{}".format(epoch , iou_disc, iou_cup, delta_cdr))
        if iou_cup > self.best_metrics['cup'] or iou_disc > self.best_metrics['disc'] or delta_cdr < self.best_metrics['delta_cdr'] :
            if iou_cup > self.best_metrics['cup']:
                self.best_metrics['cup'] = iou_cup
            if iou_disc > self.best_metrics['disc']:
                self.best_metrics['disc'] = iou_disc
            if delta_cdr < self.best_metrics['delta_cdr']:
                self.best_metrics['delta_cdr'] = delta_cdr

            print("a best model saved with iou for cup {} and for disc is {} on epoch {}".format(iou_cup, iou_disc, epoch))
            self.save_model(epoch, iou_disc, iou_cup, delta_cdr, timestampLaunch, args.hyparams_dict, save_dir=args.output_dir)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=list, default=['origa', 'refuge', 'drishti'],
                        choices=['origa', 'refuge', 'dristhi'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--source1_dataset', type=str, default='/storage/zwang/datasets/origa/split_ORIGA/',
                        help='dataset name (default: pascal)')
    parser.add_argument('--source2_dataset', type=str, default='/storage/zwang/datasets/refuge/split_refuge/',
                        help='dataset name (default: pascal)')
    parser.add_argument('--target_dataset', type=str, default='/storage/zwang/datasets/drishti/split_drishti/',
                        help='dataset name (default: pascal)')
    parser.add_argument('--Lf', type=float, default=2.0, metavar='LF',
                        help='gradient penalty (default: auto)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        metavar='w', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')

    parser.add_argument('--Cs', action='store_true', default=True, help='if use sample proportion prior')

    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--loss-type', type=str, default='bce',
                        choices=['ce', 'focal', 'bce'],
                        help='loss func type (default: ce)')
    parser.add_argument('--lr_critic', type=float, default=1e-4,
                        help='skip validation during training')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='skip validation during training')
    parser.add_argument('--lambda_g', type=float, default=1,
                        help='skip validation during training')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: auto)')

    parser.add_argument('--k_disc', type=int, default=1,
                        help='disc step')
    parser.add_argument('--k_src', type=int, default=1,
                        help='src step')
    parser.add_argument('--k_targ', type=int, default=1,
                        help='skip validation during training')
    # checking point
    parser.add_argument('--resume', type=str, default= "pretrained/deeplab-resnet.pth.tar", #'best_origa/m-adda_wgan_clip_0.03v_9.8.pth.tar',#'pretrained/deeplab-resnet.pth.tar',#"m-adda_wganv_9.1.pth.tar", #"run/glaucoma/best_experiment_2.pth.tar",
        help='put the path to resuming file if needed')
    parser.add_argument('--mu', type = float, default=0.01, help="balancing paramter")
    parser.add_argument('--output-dir', type=str, default='multi_wgan_clip_0.03', help='output path')

    parser.add_argument('--save_model', type=bool, default= 'False',#"m-adda_wganv_9.1.pth.tar", #"run/glaucoma/best_experiment_2.pth.tar",
                        help='put the path to resuming file if needed')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    args.batch_size = 8
    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'glaucoma': 0.007,
        }
    args.lr = 1e-4 # 5e-5 best model
    args.hyparams_dict = {'lr_critic': args.lr_critic , 'lr_gen': args.lr ,'Lf': args.Lf , 'gamma': args.gamma ,'mu': args.mu, 'k_disc': args.k_disc, 'k_src': args.k_src}
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    trainer = multi_source(args)
    trainer.train_meta_source(args)

main()
with open('dual.o', 'w') as f:
    f.close()





















# import pdb
# import argparse
# import os
# import numpy as np
# import random
# from tqdm import tqdm, trange
# from PIL import Image
# import matplotlib.pyplot as plt
# from datasets import make_data_loader
# from utils.metrics import compute_iou
# from utils.test_sanity import sanity_check,sanity_check_2
# from trainer_dual_source.trainer_pretrain_meta import multisource_metatrainer

# import torch
# class multi_source:
#     def __init__(self, args):
#         kwargs = {'num_workers': 4, 'pin_memory': True}
#         self.num_class = 2
#         self.num_domains = 2
#         self.srca_train_loader, self.srca__val_loader, self.srca__test_loader, self.nclass = make_data_loader(0, args.source1_dataset, args, **kwargs)
#         self.srcb_train_loader,self.srcb_val_loader, self.srcb_test_loader, self.nclass = make_data_loader(1, args.source2_dataset, args, **kwargs)
#         self.tgt_train_loader, self.tgt_val_loader, self.tgt_test_loader, self.nclass = make_data_loader(2, args.target_dataset, args, **kwargs)
#         self.tbar = tqdm(self.test_loader, desc='\r')
#         self.best_IoU = {'disc': 0.77, 'cup': 0.65}
#         self.attempt = 9.6
#         self.multisource_trainer = multisource_metatrainer(args, self.update_lr, self.update_gamma, self.update_step, self.num_class)
#         self.trainer_multisource(args)

#     def loop_iterable(self, iterable):
#         while True:
#             yield from iterable

#     def save_model(self, epoch, IoU):
#         print('---- MODEL SAVE ---')
#         torch.save({'epoch': epoch + 1, 'state_dict': self.multisource_trainer.generator_model.module.state_dict(), 'best_auc': str(self.best_IoU['cup']),
#                     'optimizer' : self.multisource_trainer.dda_optim.state_dict()}, 'best_origa/m-adda_wgan_clip_0.03' + "v_" + str(self.attempt) + '.pth.tar')
#         return

#     def trainer_multisource(self, args):
#         print("trainer initialized training started")
#         for epoch in range(args.epochs):
#             self.validation(args, epoch)
#             self.multisource_trainer.generator_model.train()
#             total_loss = 0
#             len_dataloader = len(self.source_loader)
#             torch.manual_seed(1 + epoch)
#             for step, source_data in enumerate(self.source_loader):
#                 image = source_data[0].cuda()
#                 mask  = source_data[1].cuda()
#                 batch_loss,_ = self.multisource_trainer.update_weights(image, mask)
#                 total_loss+=batch_loss.detach().cpu().numpy()
#                 if step %50 ==0:
#                     print('batch wise loss {} at batch {}'.format(total_loss/(step+1), step+1))
#             print("total epoch loss {}".format(total_loss/(step+1)))
#         return

#     def validation(self, args, epoch):
#         self.multisource_trainer.generator_model.eval()
#         test_loss = 0.0
#         predict_disc =None
#         target_disc = None
#         predict_cup = None
#         target_cup = None
#         for i, data in enumerate(self.tbar):
#             image, target = data[0], data[1]
#             image, target = image.cuda(), target.cuda()
#             with torch.no_grad():
#                 output,_ = self.multisource_trainer.generator_model(image)
#             test_loss = self.multisource_trainer.generator_criterion(output, target)
#             pred = output.data.cpu().numpy()
#             target = target.cpu().numpy()
#             pred[pred >= 0.5] = 1
#             pred[pred < 0.5] = 0
#             if i==0:
#                 target_disc = target[:,0,].squeeze()
#                 target_cup = target[:,1,].squeeze()
#                 predict_disc = pred[:,0,].squeeze()
#                 predict_cup= pred[:,1,].squeeze()
#             else:
#                 target_disc = np.vstack([target_disc, target[:,0,].squeeze()])
#                 target_cup = np.vstack([target_cup, target[:,1,].squeeze()])
#                 predict_disc = np.vstack([predict_disc, pred[:,0,].squeeze()])
#                 predict_cup = np.vstack([predict_cup, pred[:,1,].squeeze()])
#         #evaluator.Plot_Loss(1)
#         print('Validation on total  set of size {}'.format(len(target_disc)))
#         #print('[Epoch: %d, numImages: %5d]' % (epoch, i * args.batch_size + image.data.shape[0]))
#         # if epoch == 0:
#         #    sanity_check_2(image.detach().cpu().numpy(), target_cup, predict_cup)
#         iou_cup = compute_iou(target_cup, predict_cup)
#         iou_disc = compute_iou(target_disc, predict_disc)
#         print("for Epoch {} iou disc:{} and iou_cup:{}".format(epoch , iou_disc, iou_cup))
#         if iou_cup > self.best_IoU['cup']:
#             self.best_IoU['cup'] = iou_cup
#             self.best_IoU['disc'] = iou_disc
#             print("best iou is {} on epoch {}".format(iou_cup, epoch))
#             if args.save_model == True:
#                 self.save_model(epoch, self.best_IoU['cup'])
#                 print("best model saved at {}")


# def main():
#     parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
#     parser.add_argument('--backbone', type=str, default='resnet',
#                         choices=['resnet', 'xception', 'drn', 'mobilenet'],
#                         help='backbone name (default: resnet)')
#     parser.add_argument('--dataset', type=str, default=['origa', 'refuge', 'drishti'],
#                         choices=['origa', 'drishti', 'refuge'],
#                         help='dataset name (default: pascal)')
#     parser.add_argument('--source1_dataset', type=str, default='/storage/shreya/datasets/origa/split_ORIGA/',
#                         help='dataset name (default: pascal)')
#     parser.add_argument('--source2_dataset', type=str, default='/storage/shreya/datasets/refuge/split_refuge/',
#                         help='dataset name (default: pascal)')
#     parser.add_argument('--target_dataset', type=str, default='/storage/shreya/datasets/drishti/split_drishti/',
#                         help='dataset name (default: pascal)')

#     parser.add_argument('--lr', type=float, default=None, metavar='LR',
#                         help='learning rate (default: auto)')
#     parser.add_argument('--lr-scheduler', type=str, default='poly',
#                         choices=['poly', 'step', 'cos'],
#                         help='lr scheduler mode: (default: poly)')
#     parser.add_argument('--momentum', type=float, default=0.9,
#                         metavar='M', help='momentum (default: 0.9)')
#     parser.add_argument('--weight-decay', type=float, default=1e-5,
#                         metavar='M', help='w-decay (default: 5e-4)')
#     parser.add_argument('--nesterov', action='store_true', default=False,
#                         help='whether use nesterov (default: False)')
#     # cuda, seed and logging
#     parser.add_argument('--no-cuda', action='store_true', default=
#                         False, help='disables CUDA training')
#     parser.add_argument('--gpu-ids', type=str, default='0',
#                         help='use which gpu to train, must be a \
#                         comma-separated list of integers only (default=0)')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')

#     parser.add_argument('--loss-type', type=str, default='bce',
#                         choices=['ce', 'focal', 'bce'],
#                         help='loss func type (default: ce)')
#     parser.add_argument('--lr_critic', type=int, default=1e-4,
#                         help='skip validation during training')
#     parser.add_argument('--gamma', type=int, default=10,
#                         help='skip validation during training')
#     parser.add_argument('--lambda_g', type=int, default=1,
#                         help='skip validation during training')
#     parser.add_argument('--epochs', type=int, default=400, metavar='N',
#                         help='number of epochs to train (default: auto)')

#     parser.add_argument('--k_disc', type=int, default=1,
#                         help='skip validation during training')
#     parser.add_argument('--k_src', type=int, default=1,
#                         help='skip validation during training')
#     parser.add_argument('--k_targ', type=int, default=1,
#                         help='skip validation during training')
#     # checking point
# <<<<<<< Updated upstream:trainer_dual_source/source_only.py
#     parser.add_argument('--resume', type=str, default= None, #'best_origa/m-adda_wgan_clip_0.03v_9.8.pth.tar',#"m-adda_wganv_9.1.pth.tar", #"run/glaucoma/best_experiment_2.pth.tar",
# =======
#     parser.add_argument('--resume', type=str, default= 'best_origa/m-adda_wgan_clip_0.03v_9.8.pth.tar',#"m-adda_wganv_9.1.pth.tar", #"run/glaucoma/best_experiment_2.pth.tar",
# >>>>>>> Stashed changes:trainer_dual_source/train_source_only.py
#                         help='put the path to resuming file if needed')
#     parser.add_argument('--save_model', type=bool, default= 'False',#"m-adda_wganv_9.1.pth.tar", #"run/glaucoma/best_experiment_2.pth.tar",
#                         help='put the path to resuming file if needed')
#     args = parser.parse_args()
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#     if args.cuda:
#         try:
#             args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
#         except ValueError:
#             raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
#     args.batch_size = 2
#     if args.lr is None:
#         lrs = {
#             'coco': 0.1,
#             'cityscapes': 0.01,
#             'pascal': 0.007,
#             'glaucoma': 0.007,
#         }
# <<<<<<< Updated upstream:trainer_dual_source/source_only.py
#     args.lr = 5e-5 # 5e-5 best model
# =======
#     args.lr = 1e-5 # 5e-5 best model
# >>>>>>> Stashed changes:trainer_dual_source/train_source_only.py
#     torch.manual_seed(1)
#     torch.cuda.manual_seed(1)
#     np.random.seed(1)
#     random.seed(1)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.benchmark = False
#     multi_source(args)

# main()
