import os
import torch
import pdb
import numpy as np
from datasets.helper_func import read_dataset
from datasets.preprocessor import Preprocessor

MASK_PATHS = {"origa": "/storage/zwang/datasets/origa/",
              "drishti":"/storage/zwang/datasets/drishti/Disc_Cup_Masks/",
              "refuge": "/storage/zwang/datasets/refuge/cropped/Disc_Cup_Masks/"}
class make_dataset:
    NUM_CLASSES = 2
    def __init__(self, args, split, dataset, multi_source_type):
        super(make_dataset, self).__init__()
        split_path = {'train': 'train_shuffled_data.txt', 'test': 'test_shuffled_data.txt', 'combined': 'all_shuffled_data.txt'}
        self.source = {0: None, 1: None}
        if split == 'train':
            self.source[0] = read_dataset(split_path[split],
                                          args.source1_dataset,
                                          MASK_PATHS[dataset[0]],
                                          dataset[0])
            self.source[1] = read_dataset(split_path[split], args.source2_dataset,MASK_PATHS[dataset[1]], dataset[1])
            self.min_len_source_dataset = min([len(self.source[0]), len(self.source[1])])
            self.min_source_index = 0 if len(self.source[0]) < len(self.source[1]) else 1
        self.target =  read_dataset(split_path['combined'], args.target_dataset, MASK_PATHS[dataset[2]], dataset[2])
        self.split = split
        self.multi_source_type = multi_source_type
        self.preprocessor = Preprocessor(prep_method=['transform_image'], resize= 600, crop=(530,530), options={'norm':[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]})

    def __getitem__(self, index):
        if self.split == 'test' and self.multi_source_type == 'twosource':
            target_mask = self.preprocessor.preprocess_image(self.target[index][1], 'mask')
            target_image = self.preprocessor.preprocess_image(self.target[index][0], 'image')
            return target_image, target_mask
        if self.multi_source_type == 'pretrain':
            mask = self.preprocessor.preprocess_image(self.source1[index][1], 'mask')
            image = self.preprocessor.preprocess_image(self.source1[index][0], 'image')
            return image, mask
        elif self.multi_source_type == 'single' :
            s1_mask = self.preprocessor.preprocess_image(self.source[0][index][1], 'mask')
            s1_image = self.preprocessor.preprocess_image(self.source[0][index][0], 'image')
            t1_mask = self.preprocessor.preprocess_image(self.target[index][1], 'mask')
            t1_image = self.preprocessor.preprocess_image(self.target[index][0], 'image')
            return s1_image, s1_mask, t1_image, t1_mask
        else:
            source_image = None
            source_mask = None
            index_other = index
            if index > self.min_len_source_dataset-1:
                index_other = index % self.min_len_source_dataset
            # To-do change this : slighltly complicated logic to order the sources in the increasing order of their lengths
            for ind, sour in enumerate([self.min_source_index, abs(self.min_source_index%1)]):
                if ind == 0:
                    source_image = self.preprocessor.preprocess_image(self.source[sour][index_other][0], 'image')
                    source_mask = self.preprocessor.preprocess_image(self.source[sour][index_other][1], 'mask')
                    continue
                source_image = torch.stack((source_image, self.preprocessor.preprocess_image(self.source[sour][index][0], 'image')),0)
                source_mask = np.stack([source_mask, self.preprocessor.preprocess_image(self.source[sour][index][1], 'mask')], 0)
            assert source_image.shape[0] == 2
            assert source_mask.shape[0] == 2
            return source_image, source_mask[0]

    def __len__(self):
        if self.split == 'test':
            return len(self.target)
        if self.multi_source_type == 'single':
            return len(self.source[0])
        else:
            return max(len(self.source[0]), len(self.source[1]))

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
