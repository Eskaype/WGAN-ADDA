import os
import torch
import pdb
import cv2
import numpy as np
from datasets.helper_func import read_dataset
from datasets.preprocessor import Preprocessor
#from datasets.preprocessor import randomHorizontalFlip, randomVerticleFlip, randomRotate90, randomHueSaturationValue, randomShiftScaleRotate

MASK_PATHS = {"origa": "/storage/shreya/datasets/origa/",
              "drishti":"/storage/shreya/datasets/drishti/Disc_Cup_Masks/",
              "refuge": "/storage/shreya/datasets/refuge/cropped/Disc_Cup_Masks/"}


class make_new_dataset:
    NUM_CLASSES = 2
    def __init__(self, args, split, dataset, data_path, multi_source_type):
        split_path = {'train': 'train_shuffled_data.txt', 'test': 'test_shuffled_data.txt', 'combined': 'all_shuffled_data.txt'}
        self.dataset = dataset
        self.source = read_dataset(split_path[split], data_path, MASK_PATHS[dataset], dataset)
        self.preprocessor = Preprocessor(prep_method=['transform_image'], resize= 450, crop=(400,400), options={'norm':[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]})
        # if dataset == 'drishti':
        #     import pdb
        #     pdb.set_trace()
        #     source_paths = self.data_augment(self.source)
        #     self.source.append(source_paths)

    def __getitem__(self, index):
        source_image = None
        source_mask = None
        source_image = self.preprocessor.preprocess_image(self.source[index][0], 'image', self.dataset)
        source_mask = self.preprocessor.preprocess_image(self.source[index][1], 'mask', self.dataset)
        #print(source_image.shape, source_mask.shape)
        return source_image, source_mask

    def __len__(self):
        return len(self.source)

    def data_augment(self, source):
        source_aug = []
        for index, source_path in enumerate(source):
            image_path = source_path[0]
            mask_path = source_path[1]
            # image = self.preprocessor.randomHueSaturationValue(image,
            #                         hue_shift_limit=(-30, 30),
            #                         sat_shift_limit=(-5, 5),
            #                         val_shift_limit=(-15, 15))

            # image, mask = self.preprocessor.randomShiftScaleRotate(image, mask,
            #                                 shift_limit=(-0.1, 0.1),
            #                                 scale_limit=(-0.1, 0.1),
            #                                 aspect_limit=(-0.1, 0.1),
            #                                 rotate_limit=(-0, 0))
            img_path_pref = image_path.split('.jpg')[0]
            msk_path_pref = mask_path.split('.bmp')[0]
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, 0)
            #mask = self.preprocessor.read_mask_augmentation(mask)
            image, mask = self.preprocessor.randomHorizontalFlip(image, mask, 0.2)
            image, mask = self.preprocessor.randomVerticleFlip(image, mask, 0.2)
            image, mask = self.preprocessor.randomRotate90(image, mask, 0.2)
            result = cv2.imwrite(img_path_pref+'_aug.jpg', image)
            assert result == True
            result = cv2.imwrite(msk_path_pref+'_aug.bmp', mask)
            assert result == True
            source_aug.append([ img_path_pref+'_aug.jpg',msk_path_pref+'_aug.bmp'])
        return source_aug

class make_dataset:
    NUM_CLASSES = 2
    def __init__(self, args, split, dataset, multi_source_type):
        super(make_dataset, self).__init__()
        split_path = {'train': 'train_shuffled_data.txt', 'test': 'test_shuffled_data.txt', 'combined': 'all_shuffled_data.txt'}
        self.dataset = dataset
        self.source = {0: None, 1: None}
        if split == 'train':
            self.source[0] = read_dataset(split_path[split],
                                          args.source1_dataset,
                                          MASK_PATHS[dataset[0]],
                                          dataset[0])
            if multi_source_type == 'twosource':
                self.source[1] = read_dataset(split_path[split], args.source2_dataset,MASK_PATHS[dataset[1]], dataset[1])
                self.min_len_source_dataset = min([len(self.source[0]), len(self.source[1])])
                self.min_source_index = 0 if len(self.source[0]) < len(self.source[1]) else 1
        if dataset[-1] == 'all':
            self.target =  read_dataset(split_path['combined'], args.target_dataset, MASK_PATHS[dataset[0]], dataset[0])
            self.target.append(read_dataset(split_path['combined'], args.target_dataset, MASK_PATHS[dataset[1]], dataset[1]))
        elif split == 'test' and multi_source_type=='pretrain':
            self.target = read_dataset(split_path['test'],  args.source1_dataset, MASK_PATHS[dataset[0]], dataset[0])
            self.target.append(read_dataset(split_path['test'],  args.source2_dataset, MASK_PATHS[dataset[1]], dataset[1]))
        elif split == 'test' and multi_source_type != 'pretrain':
            self.target = read_dataset(split_path['test'], args.target_dataset, MASK_PATHS[dataset[-1]], dataset[-1])
        else:
            self.target =  read_dataset(split_path['train'], args.target_dataset, MASK_PATHS[dataset[-1]], dataset[-1])
            #print("dataset split {} {} {}".format(args.target_dataset,  MASK_PATHS[dataset[-1]],  dataset[-1]))
        self.split = split
        self.multi_source_type = multi_source_type
        self.preprocessor = Preprocessor(prep_method=['transform_image'], resize= 450, crop=(400,400), options={'norm':[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]})

    def __getitem__(self, index):
        if self.split == 'test' and (self.multi_source_type == 'twosource' or self.multi_source_type == 'single'):
            target_mask = self.preprocessor.preprocess_image(self.target[index][1], 'mask', self.dataset[-1])
            target_image = self.preprocessor.preprocess_image(self.target[index][0], 'image', self.dataset[-1])
            return target_image, target_mask
        if self.multi_source_type == 'pretrain' and self.split == 'test':
            mask = self.preprocessor.preprocess_image(self.source[1][index][1], 'mask')
            image = self.preprocessor.preprocess_image(self.source[1][index][0], 'image')
            return image, mask
        elif self.multi_source_type == 'single' :
            s1_mask = self.preprocessor.preprocess_image(self.source[0][index][1], 'mask', self.dataset[0])
            s1_image = self.preprocessor.preprocess_image(self.source[0][index][0], 'image', self.dataset[0])
            return s1_image, s1_mask
        else:
            source_image = None
            source_mask = None
            index_other = index
            if index > self.min_len_source_dataset-1:
                index_other = index % self.min_len_source_dataset
            # To-do change this : slighltly complicated logic to order the sources in the increasing order of their lengths
            for ind, sour in enumerate([self.min_source_index, abs(self.min_source_index%1)]):
                if ind == 0:
                    source_image = self.preprocessor.preprocess_image(self.source[sour][index_other][0], 'image', self.dataset[sour])
                    source_mask = self.preprocessor.preprocess_image(self.source[sour][index_other][1], 'mask', self.dataset[sour])
                    continue
                source_image = torch.stack((source_image, self.preprocessor.preprocess_image(self.source[sour][index][0], 'image', self.dataset[sour])),0)
                source_mask = np.stack([source_mask, self.preprocessor.preprocess_image(self.source[sour][index][1], 'mask', self.dataset[sour])], 0)
            assert source_image.shape[0] == 2
            assert source_mask.shape[0] == 2
            return source_image, source_mask

    def __len__(self):
        if self.split == 'test':
            return len(self.target)
        if self.multi_source_type == 'single':
            return len(self.source[0])
        else:
            #return max(len(self.source[0]), len(self.source[1]))
            return len(self.source[0]), len(self.source[1])


    def data_augment(self):
        _img = None
        _msk = None
        for ite, img_ in enumerate(self.img_):
            msk_ = self.msk_[ite]
            if ite == 0:
                _img = np.repeat(img_[np.newaxis, :,:,:], 2, axis=0)
                _msk = np.repeat(msk_[np.newaxis, :,:,:], 2, axis=0)
                continue
            _img = np.vstack((_img, np.repeat(img_[np.newaxis, :,:,:], 2, axis=0)))
            _msk = np.vstack((_msk, np.repeat(msk_[np.newaxis, :,:,:], 2, axis=0)))
        self.img_ = np.array(_img)
