import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from utils.test_sanity import check_preprocess_sanity

class Preprocessor:
    def __init__(self, prep_method:str, resize: tuple, crop: tuple, options:dict):
        self.resize = resize
        self.crop = crop
        self.preprocess_image_methods = prep_method
        transforms_list = []
        transforms_list.extend([transforms.ToTensor(),
                               transforms.Normalize(options['norm'][0], options['norm'][1])])

        self.transform_sequence = transforms.Compose(transforms_list)

    def center_crop(self, img: 'Image', crop_size: tuple, num_channels: int):
        if num_channels == 2:
            w, h = img.shape
        else:
            w, h, c = img.shape
        tw, th = crop_size
        left = int((w - tw) / 2)
        top = int((h - th) / 2)
        right = int((w + tw) / 2)
        bottom = int((h + th) / 2)
        if num_channels == 2:
            img = img[left:right, top:bottom]
        else:
            img = img[left:right, top:bottom, :]
        return img

    def transform_image(self, bgr, resize_width=None, resize_height=None, clip=2.0):
        """
        CLAHE and resize
        """
        bgr = cv2.resize(bgr, (self.resize, self.resize),  interpolation =cv2.INTER_NEAREST)
        bgr = self.center_crop(bgr, self.crop, 3)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        image  = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image

    def read_image(self, org_msk):
        img_h = 800
        img_w = 800
        mask_d = np.zeros((org_msk.shape[0], org_msk.shape[1]))
        mask_c = np.zeros((org_msk.shape[0], org_msk.shape[1]))
        mask_d[org_msk==127] = 1 # larger box grey: disc
        mask_d[org_msk==255] = 1 # disc should cover cup
        mask_c[org_msk==255] = 1 # smaller box black: cup
        mskc = cv2.resize(mask_c, (img_w, img_h))
        mskd = cv2.resize(mask_d, (img_w, img_h))
        img = np.zeros((img_w, img_h))
        img.fill(255)
        img[mskd==1] = 128
        img[mskc==1] = 0
        return img

    def msk_to_msk(self, org_msk):
        msk_oc = np.zeros((org_msk.shape[0],org_msk.shape[1]))
        msk_oc[org_msk==0]=1
        msk_od = np.zeros((org_msk.shape[0],org_msk.shape[1]))
        msk_od[org_msk==0]=1
        msk_od[org_msk==128]=1
        msk = np.stack([msk_od, msk_oc], axis=2)
        return msk

    def preprocess_image(self, file_name: str, image_type: 'mask'):
        """
            self.preprocess_image_methods = ['centre_crop', 'clahe_norm', 'adjust_gamma']
        """
        if image_type == 'mask':
            org_msk = cv2.imread(file_name, 0)
            org_msk = self.read_image(org_msk)
            org_msk = cv2.resize(org_msk, (self.resize, self.resize), interpolation =cv2.INTER_NEAREST)
            org_msk = self.center_crop(org_msk, self.crop, 2)
            org_msk = self.msk_to_msk(org_msk)
            image = org_msk.transpose((2,0,1))
            ## assert check_preprocess_sanity(image) == True
        else:
            for action in self.preprocess_image_methods:
                try:
                    image = cv2.imread(file_name)
                    image = getattr(Preprocessor, action)(self, image)
                except AttributeError:
                    raise ValueError('Preprocessor {} not implemented.'.format(action))
                # convert numpy to Image
                image = Image.fromarray(image)
                image = self.transform_sequence(image)
        return image
