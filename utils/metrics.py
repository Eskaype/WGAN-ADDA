import numpy as np
import os
import cv2
from sklearn.metrics import mean_absolute_error
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
    iou = sum(iou_mask)/len(iou_mask)
    dod = 2.*(iou)/(iou+1.)
    return iou, dod

def elp(maps):
    gray1 = np.zeros(maps.shape)
    gray1 = np.array(gray1, dtype=np.uint8)
    gray1[maps==1] = 128
    _, thresh1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contour1, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area1 = [cv2.contourArea(cnt) for cnt in contour1]
    #print(area1)
    area2 = [a if a < 100000. else 0. for a in area1]
    contour = contour1[np.argmax(np.array(area2))]
    #print(contour)
    ellipse = cv2.fitEllipse(contour)
    poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
            (int(ellipse[1][0]/2), int(ellipse[1][1]/2)), int(ellipse[2]), 0, 360, 5)
    pmask = np.zeros(maps.shape)
    cv2.fillPoly(pmask,[poly],1)
    return ellipse, pmask

def compute_cdr(predict_cup, predict_disc, target_cup, target_disc):
    CDR = []
    TCDR = []
    for j in range(len(predict_cup)):
        eDmap,De_disc_map = elp(predict_disc[j])
        tDmap,test_msk_disc = elp(target_disc[j])
        eCmap,De_cup_map = elp(predict_cup[j])
        tCmap,test_msk_cup = elp(target_cup[j])
        CDR.append(eCmap[1][1]/eDmap[1][1])
        TCDR.append(tCmap[1][1]/tDmap[1][1])
        #plot_GT(test_msk_cup, test_msk_disc, 2*i+j, False)
        #plot_GT(De_cup_map, De_disc_map, 2*i+j, name, True)
    delta = mean_absolute_error(TCDR, CDR)
    return delta