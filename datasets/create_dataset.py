#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.io as sio
import scipy.misc
#from tensorflow.keras.preprocessing import image
from skimage.transform import rotate, resize
from skimage.measure import label, regionprops
from time import time
#from utils import pro_process, BW_img, disc_crop
from PIL import Image
from matplotlib.pyplot import imshow
#import keras
import cv2
import os
import random

import pickle

import argparse


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
IMG_SIZE = 512
ch_mean = np.array([40.434, 70.556, 114.224])

def img_scale(img, scale):
    d = img.shape[0]
    off = int((d-d/scale)/2)
    imgcrop = img[off:(d-off),off:(d-off)]
    return cv2.resize(imgcrop, (d, d))


def img_rotate(img, degree):
    num_rows,num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols//2, num_rows//2), degree, 1)
    img = cv2.warpAffine(img,rotation_matrix,(num_cols,num_rows))
    return img

def transform_image(bgr, resize_width=None, resize_height=None, clip=2.0):
    """
    CLAHE and resize
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image  = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image

DiscROI_size = 800
DiscSeg_size = 640
CDRSeg_size = 400
#CDRSeg_model = MNetModel.DeepModel(size_set=CDRSeg_size)
#CDRSeg_model.load_weights('Model_MNet_ORIGA_pretrain.h5')



def msk_to_msk(org_msk):
    msk_oc = np.zeros((org_msk.shape[0],org_msk.shape[1]))
    msk_oc[org_msk==0]=1
    msk_od = np.zeros((org_msk.shape[0],org_msk.shape[1]))
    msk_od[org_msk==0]=1
    msk_od[org_msk==128]=1
    msk = np.stack([msk_od, msk_oc], axis=2)
    return msk

def pro_process_msk(img,input_size):
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (input_size, input_size))
    img = np.asarray(img).astype('float32')
    return img

with open('../../data/cropped/f2/train.p', 'rb') as f:
    trainll = pickle.load(f)
with open('../../data/cropped/f2/validation.p', 'rb') as f:
    vall = pickle.load(f)
with open('../../data/cropped/f2/test.p', 'rb') as f:
    testl = pickle.load(f)

trainl = trainll+vall+testl
print(len(trainl))
tl = [os.path.basename(t) for t in trainl]


data_type = '.jpg'
# data_img_path = './train_img/'
data_img_path1 = "../../data/cropped/keras/Glaucoma/"
data_img_path2 = "../../data/cropped/keras/Non-Glaucoma/"

data_msk_path1 = "../../data/cropped/Disc_Cup_Masks/keras/Glaucoma/"
data_msk_path2 = "../../data/cropped/Disc_Cup_Masks/keras/Non-Glaucoma/"

file_list1 = [file for file in os.listdir(data_img_path1) if file.lower().endswith(data_type)]
file_list2 = [file for file in os.listdir(data_img_path2) if file.lower().endswith(data_type)]
print(str(len(file_list1)))
print(str(len(file_list2)))

train_img = []
train_msk = []
val_img = []
val_msk = []

##### remove rescaling and polar transformer
for lineIdx in range(0, len(file_list1)):
    temp_txt = [elt.strip() for elt in file_list1[lineIdx].split(',')]
    if temp_txt[0] in tl:
        for clip in [8.0]:
            for flip in [None]:
                for rotated in [None]:
                    for scale in [None]:
                    #for scale in [None]:
                        print(' Processing Img: ' + temp_txt[0])
                        msk_txt = temp_txt[0][:-4] + '.bmp'
                        # load image
                        org_img = np.asarray(Image.open(data_img_path1 + temp_txt[0]).resize((IMG_SIZE,IMG_SIZE), resample=Image.NEAREST))
                        org_msk = cv2.imread(data_msk_path1+msk_txt, 0)
                        org_msk = cv2.resize(org_msk, (IMG_SIZE, IMG_SIZE), interpolation =cv2.INTER_NEAREST)
                        print("unique",np.unique(org_msk))
                        tran_msk = msk_to_msk(org_msk)
                        if clip != None:
                            org_img = transform_image(org_img, clip=clip)
                        if scale != None:
                            org_img = img_scale(org_img, scale=scale)
                            tran_msk = np.array(img_scale(np.array(tran_msk,dtype=np.uint8), scale=scale), dtype=np.float64)
                        train_img.append(org_img)
                        train_msk.append(tran_msk)

for lineIdx in range(0, len(file_list2)):
    temp_txt = [elt.strip() for elt in file_list2[lineIdx].split(',')]
    if temp_txt[0] in tl:
        for clip in [8.0]:
            for flip in [None]:
                for rotated in [None]:
                    for scale in [None]:
                    #for scale in [None]:
                        print(' Processing Img: ' + temp_txt[0])
                        msk_txt = temp_txt[0][:-4] + '.bmp'
                        # load image
                        org_img = np.asarray(Image.open(data_img_path2 + temp_txt[0]).resize((IMG_SIZE,IMG_SIZE), resample=Image.NEAREST))
                        org_msk = cv2.imread(data_msk_path2+msk_txt, 0)
                        org_msk = cv2.resize(org_msk, (IMG_SIZE, IMG_SIZE), interpolation =cv2.INTER_NEAREST)
                        print("unique",np.unique(org_msk))
                        tran_msk = msk_to_msk(org_msk)
                        if clip != None:
                            org_img = transform_image(org_img, clip=clip)
                        if scale != None:
                            org_img = img_scale(org_img, scale=scale)
                            tran_msk = np.array(img_scale(np.array(tran_msk,dtype=np.uint8), scale=scale), dtype=np.float64)
                        train_img.append(org_img)
                        train_msk.append(tran_msk)

idorder = list(range(len(train_img)))
random.shuffle(idorder)
train_imgs = [train_img[i] for i in idorder]
train_msks = [train_msk[i] for i in idorder]
import pdb
train_img = np.stack(train_imgs, axis=0)
train_msk = np.stack(train_msks, axis=0)

print(train_img.shape)
print(train_msk.shape)

#test data for GS
data_type = '.jpg'
gsdata_img_path1 = "../../data/croppedGS/image/keras/"
gsdata_msk_path1 = "../../data/croppedGS/Disc_Cup_Masks/keras/"
gsfile_list1 = [file for file in os.listdir(gsdata_img_path1) if file.lower().endswith(data_type)]
print(str(len(gsfile_list1)))
test_img = []
test_msk = []
test_org = []
for lineIdx in range(0, len(gsfile_list1)):
    temp_txt = [elt.strip() for elt in gsfile_list1[lineIdx].split(',')]
    print(' Processing Img: ' + temp_txt[0])
    msk_txt = temp_txt[0][:-4] + '.bmp'
    # load image
    for clip in [None]:
        for scale in [None]:
            org_img = np.asarray(Image.open(gsdata_img_path1 + temp_txt[0]).resize((IMG_SIZE,IMG_SIZE), resample=Image.NEAREST))
            org_msk = cv2.imread(gsdata_msk_path1+msk_txt, 0)
            org_msk = cv2.resize(org_msk, (IMG_SIZE, IMG_SIZE), interpolation =cv2.INTER_NEAREST)
            if clip != None:
                org_img = transform_image(org_img, clip=clip)
            if scale != None:
                org_img = img_scale(org_img, scale = scale)
            tran_msk = msk_to_msk(org_msk)
            if scale !=None:
                tran_msk = np.array(img_scale(np.array(tran_msk,dtype=np.uint8), scale=scale), dtype=np.float64)
            print("unique",np.unique(org_msk))
            orig_msk = tran_msk
            test_img.append(org_img)
            test_msk.append(tran_msk)
            test_org.append(orig_msk)


test_img = np.stack(test_img, axis=0)
test_msk = np.stack(test_msk, axis=0)
test_org = np.stack(test_org, axis=0)
# #####
np.savez('data_aug_msk_512', train_img=train_img, train_msk=train_msk, test_img=test_img, test_msk=test_msk, test_org=test_org)
exit()
data = np.load('../../data/data_aug_msk_512_path.npz')
train_img = data['train_img'] # source images: N_src x W x H x C
train_msk = data['train_msk'] # source masks: N_src x W x H x C
test_img = data['test_img']# target images: N_tar x W x H x C
test_msk = data['test_msk'] # target masks: N_tar x W x H x C

#compute Mean and std for train
print(train_img.shape)
mean_ = [train_img[:,:,:,0].mean(), train_img[:,:,:,1].mean(), train_img[:,:,:,2].mean()]
stdev = [train_img[:,:,:,0].std(), train_img[:,:,:,1].std(), train_img[:,:,:,2].std()]
'''Mean [171.20852008007813 63.7885528464272 113.94485663867188]'''
''' std 53.84353997347984 80.85453641601562 46.345766491868424 '''

'''Mean 130.13397167388615 61.90945224319307 18.512967202970298
std 43.31505632159788 28.50909879194026 14.385093303758374 '''
train_img[:,:,:,0] = train_img[:,:,:,0] - mean_[0]
train_img[:,:,:,1] = train_img[:,:,:,1] - mean_[1]
train_img[:,:,:,2] = train_img[:,:,:,2] - mean_[2]

test_img = (test_img-train_img.mean(axis=(0,1,2), keepdims=True))/train_img.std(axis=(0,1,2), keepdims=True)
train_img = (train_img-train_img.mean(axis=(0,1,2), keepdims=True))/train_img.std(axis=(0,1,2), keepdims=True)


print(test_img.mean(), test_img.std())
print(train_img.mean(), train_img.std())
#= test_img/np.max(test_img)
np.savez('data_noaug_norm_msknorez', train_img=train_img, train_msk=data['train_msk'], test_img=test_img, test_msk=data['test_msk'])
# Write one train, test image and msk
# print(train_msk.shape)
# cv2.imwrite('train-img.jpg', train_img[0])
# train_msk[0,:,:,1][train_msk[0,:,:,1]==1]=255
# img = train_msk[0,:,:,0]*0.5 + train_msk[0,:,:,1]*0.5
# cv2.imwrite('train-msk.jpg', img.astype(int))
# cv2.imwrite('test-img.jpg', test_img[0])
# #Image.imwrite('test-msk.jpg', train_msk[0],0)
