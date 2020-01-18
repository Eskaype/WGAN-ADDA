import os
import torch
import matplotlib.pyplot as plt

class visualize_plot(object):
    def __init__(self, exp_name):
        self.gen_loss = []
        self.disc_loss = []
        self.train_loss = []
        self.test_loss = []
        self.mIoU_disc = []
        self.mIoU_cup = []
        self.exp_name = exp_name

    def update_loss(self, loss1, loss2, mIoU, epoch):
        self.gen_loss.append(loss1)
        self.disc_loss.append(loss2)
        self.mIoU_disc.append(mIoU['disc']*100)
        self.mIoU_cup.append(mIoU['cup']*100)
        self.epoch = epoch
        self.visualize_image(self.exp_name)
        return

    def visualize_image(self, exp_name ):
        plt.figure()
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        plt.plot([i for i  in range(self.epoch)], self.gen_loss, 'r')
        plt.plot([i for i in range(self.epoch)], self.disc_loss, 'b')
        fig.savefig('WGAN_loss' + exp_name+ '.png')   # save the figure to file
        plt.close(fig)
        plt.figure()
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        plt.plot([i for i  in range(self.epoch)], self.mIoU_disc, 'r')
        plt.plot([i for i  in range(self.epoch)], self.mIoU_cup, 'c')
        fig.savefig('WGAN_IOU'+ exp_name+ '.png')   # save the figure to file
        plt.close(fig)
        return
    def visualize_loss(self, exp_name ):
        plt.figure()
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        plt.plot([i for i  in range(self.epoch)], self.train_loss, 'r')
        plt.plot([i for i in range(self.epoch)], self.test_loss, 'b')
        fig.savefig('GAN_loss.png')   # save the figure to file
        plt.close(fig)
        return