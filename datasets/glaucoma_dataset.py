import os
import torch

from datasets.helper_func import read_dataset
from datasets.preprocessor import Preprocessor


class make_dataset:
    NUM_CLASSES = 2
    def __init__(self, args, split='train'):
        super(make_dataset, self).__init__()
        split_path = {'train':'Origa_shuffled_train_images.txt' , 'test':'Origa_shuffled_test_images.txt'}
        self.source[0] = read_dataset(split_path[split],args.source1_dataset )
        self.source[1] = read_dataset(split_path[split], args.source2_dataset)
        self.target =  read_dataset(split_path[split], args.target_dataset)
        self.split = split
        self.multi_source_type = 'single'
        self.num_sources = 1
        self.min_len_source_dataset = min([len(self.source[0]), len(self.source[1])])
        self.min_source_index = 0 if len(self.source[0]) < len(self.source[1]) else 1
        self.preprocessor = Preprocessor(prep_method=['transform_image'], resize= 600, crop=(530,530), options={'norm':[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]})

    def __getitem__(self, index):
        if self.multi_source_type == 'pretrain':
            mask = self.preprocessor.preprocess_image(self.source1[index][1], 'mask')
            image = self.preprocessor.preprocess_image(self.source1[index][0], 'image')
            return image, mask
        elif self.multi_source_type == 'single':
            s1_mask = self.preprocessor.preprocess_image(self.source1[index][1], 'mask')
            s1_image = self.preprocessor.preprocess_image(self.source1[index][0], 'image')
            t1_mask = self.preprocessor.preprocess_image(self.target[index][1], 'mask')
            t1_image = self.preprocessor.preprocess_image(self.target[index][0], 'image')
            return s1_image, s1_mask, t1_image, t1_mask
        else:
            source_image = None
            source_mask = None
            index_other = index
            if index > self.min_len_source_dataset:
                index_other = index % self.min_len_source_dataset
            # slighltly complicated logic to order the sources in the increasing order of their lengths
            for ind, sour in enumerate([self.min_source_index, abs(self.min_source_inde%1)]):
                if ind == 0:
                    source_image = self.preprocessor.preprocess_image(self.source[sour][index_other][0], 'image')
                    source_mask = self.preprocessor.preprocess_image(self.source[sour][index_other][1], 'mask')
                    break
                source_image.vstack(self.preprocessor.preprocess_image(self.source[sour][index][0], 'image'))
                source_mask.vstack(self.preprocessor.preprocess_image(self.source[sour][index][1], 'mask'))
            targ_mask = self.preprocessor.preprocess_image(self.target[index][1], 'mask')
            targ_image = self.preprocessor.preprocess_image(self.target[index][0], 'image')
            return sour_image, sour_mask, targ_image, targ_mask

    def __len__(self):
        if self.multi_source_type == 'single':
            return len(self.source1)
        else:
            return len(self.source1+self.source2)

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