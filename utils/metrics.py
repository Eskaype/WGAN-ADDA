import numpy as np
import os
from utils.test_sanity import sanity_check,sanity_check_2, check_preprocess_sanity, check_postprocess_sanity

def iou_msk(msk1, msk2):
    inter_vol = np.sum(np.bitwise_and(np.array(msk1,dtype=np.int32), np.array(msk2,dtype=np.int32)))
    vol1 = np.sum(msk1)
    vol2 = np.sum(msk2)
    return inter_vol/(vol1+vol2-inter_vol)

def compute_iou(target, input_):
    iou_mask  = []
    #import pdb
    #pdb.set_trace()
    #check_postprocess_sanity(target, input_, "postproc.png")
    for i in range(len(target)):
        iou_mask.append(iou_msk(input_[i], target[i]))
    return sum(iou_mask)/len(iou_mask)