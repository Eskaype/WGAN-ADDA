import cv2
import numpy as np
def sanity_check(image, target, prediction):
    target_disc = target[3,0,:,:]
    target_cup = target[3,1,:,:]
    prediction_disc = prediction[3,0,:,:]
    prediction_cup = prediction[3,1,:,:]
    target_disc[target_disc==1] = 128
    target_cup[target_cup == 1] = 127
    prediction_disc[prediction_disc==1] = 128
    prediction_cup[prediction_cup == 1] = 127
    target = target_disc+target_cup
    predict = prediction_disc+prediction_cup
    cv2.imwrite('target_disc.png', np.uint8(target_disc))
    cv2.imwrite('target_cup.png', np.uint8(target_cup))
    cv2.imwrite('prediction_disc.png', np.uint8(prediction_disc))
    cv2.imwrite('prediction_cup.png', np.uint8(prediction_cup))
    image = np.uint8(image*255)[0].transpose((1,2,0))
    cv2.imwrite('image.png', image)
    return

def sanity_check_2(image, target, prediction):
    target_disc = target[3,:,:]
    prediction_disc = prediction[3,:,:]
    target_disc[target_disc==1] = 128
    prediction_disc[prediction_disc==1] = 128
    cv2.imwrite('target_disc.png', np.uint8(target_disc))
    cv2.imwrite('prediction_disc.png', np.uint8(prediction_disc))
    image = np.uint8(image*255)[0].transpose((1,2,0))
    cv2.imwrite('image.png', image)
    return

def check_preprocess_sanity(target, save_name):
    target_disc = target[0,:,:]
    target_cup = target[1,:,:]
    proc_img = np.zeros((target_disc.shape[0],target_disc.shape[1]))
    proc_img[target_disc==1]=128
    proc_img[target_cup==1]=255
    assert list(np.unique(proc_img)) == [0.0, 128, 255]
    return cv2.imwrite(save_name, proc_img)

def check_postprocess_sanity(predict, target, save_name):
    target_disc = target[3,:,:]
    predict_disc = predict[3,:,:]
    proc_img = np.zeros((target_disc.shape[0],target_disc.shape[1]))
    proc_img[target_disc==1]=128
    proc_img[predict_disc==1]=255
    return cv2.imwrite(save_name, proc_img)