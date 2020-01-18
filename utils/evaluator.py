import numpy as np
import matplotlib.pyplot as plt

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix_disc = np.zeros((self.num_class,)*2)
        self.confusion_matrix_cup = np.zeros((self.num_class,)*2)
        self.test_loss = []
        self.train_loss = []
        self.MIoU = {'disc': [], 'cup': []}

    def Pixel_Accuracy(self, cm):
        Acc = dict()
        for i, val in enumerate(['disc', 'cup']):
            Acc[val]= np.diag(cm[i]).sum() / cm[i].sum()
        return Acc

    def Pixel_Accuracy_Class(self, cm):
        Acc = dict()
        for i, val in enumerate(['disc', 'cup']):
            Acc[val] = np.diag(cm[i]) / cm[i].sum(axis=1)
            Acc[val] = Acc[val][1]
        return Acc
    def compute_iou_custom(self, gt_image, pre_image):

        return
    def Mean_Intersection_over_Union(self, cm):
        MIoU = dict()
        for i, val in enumerate(['disc', 'cup']):
            MIoU[val]= np.diag(cm[i]) / (
                            np.sum(cm[i], axis=1) + np.sum(cm[i], axis=0) -
                            np.diag(cm[i]))
            MIoU[val] = MIoU[val][1]
            self.MIoU[val].append(MIoU[val])
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self, cm):
        FWIoU = dict()
        for i, val in enumerate(['disc', 'cup']):
            freq = np.sum(cm[i], axis=1) / np.sum(cm[i])
            iu = np.diag(cm[i]) / (
                        np.sum(cm[i], axis=1) + np.sum(cm[i], axis=0) -
                        np.diag(cm[i]))

            FWIoU[val] = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=(self.num_class)**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image, name_ = 'disc'):
        assert gt_image.shape == pre_image.shape
        #self.compute_iou_custom(gt_image[:,0,:,:], pre_image[:,0,:,:])
        self.confusion_matrix_disc+= self._generate_matrix(gt_image[:,0,:,:], pre_image[:,0,:,:])
        self.confusion_matrix_cup+= self._generate_matrix(gt_image[:,1,:,:], pre_image[:,1,:,:])

    def add_test_loss(self, loss):
        self.test_loss.append(loss)
        return
    def add_train_loss(self, loss):
        self.train_loss.append(loss)
        return
    def update_cmatrix(self):
        eval('self.confusion_matrix_'+'disc')

    def Plot_Loss(self, epoch):
        plt.figure()
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        plt.plot([i for i  in range(len(self.train_loss))], self.train_loss, 'r')
        plt.plot([i for i in range(len(self.test_loss))], self.test_loss, 'b')
        fig.savefig('GAN_loss.png')   # save the figure to file
        plt.close(fig)
        plt.figure()
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        plt.plot([i for i  in range(len(self.MIoU['disc']))], self.MIoU['disc'], 'r')
        plt.plot([i for i in range(len(self.MIoU['cup']))], self.MIoU['cup'], 'b')
        fig.savefig('GAN_IoU.png')   # save the figure to file
        plt.close(fig)
        return
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
