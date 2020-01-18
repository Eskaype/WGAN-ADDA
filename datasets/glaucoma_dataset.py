import os
import torch

from datasets.helper_func import read_dataset
from datasets.preprocessor import Preprocessor


class make_dataset:
    NUM_CLASSES = 2
    def __init__(self, args, split='train'):
        super(make_dataset, self).__init__()
        split_path = {'train':'Origa_shuffled_train_images.txt' , 'test':'Origa_shuffled_test_images.txt'}
        self.source1 = read_dataset(split_path[split],args.source1_dataset )
        self.source2 = read_dataset(split_path[split], args.source2_dataset)
        self.target =  read_dataset(split_path[split], args.target_dataset)
        self.split = split
        self.multi_source_type = 'single'
        self.preprocessor = Preprocessor(prep_method=['transform_image'], resize= 600, crop=(530,530), options={'norm':[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]})

    def __getitem__(self, index):
        mask = self.preprocessor.preprocess_image(self.source1[index][1], 'mask')
        image = self.preprocessor.preprocess_image(self.source1[index][0], 'image')
        return image, mask

    def __len__(self):
        return len(self.source1)

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