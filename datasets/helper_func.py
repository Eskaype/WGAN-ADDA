import os
import cv2

def read_dataset(file_name: str, root: str, maskpath: str, dataset: str):
    file_list = []
    with open(os.path.join(root,file_name)) as f:
        data = f.readlines()
    for line_ in data:
        file_list.append(os.path.join(line_))
    imgmaskpair = []
    for file_name in file_list:
        if dataset == 'origa':
            mask_name = file_name.split('crop')[0]
            mask_name = mask_name + 'msk_crop.bmp'
        elif dataset == 'drishti':
            mask_name = file_name.split('/')[-2:]
            mask_name = os.path.join(maskpath, mask_name.split('.')[0], '.bmp')
        else:
            mask_name = file_name.split('/')[-3:]
            mask_name = os.path.join(maskpath, mask_name.split('.')[0], '.bmp')
        imgmaskpair.append([os.path.join(root, file_name.split('\n')[0]),
                            os.path.join(root, mask_name)])
    return imgmaskpair


def test_preprocessed_image(image):
    if img_type == 'mask':
        cup = image[1]
        cup[cup==1] == 255
        disc = image[0]
        disc[disc==1] == 127
    if img_type == 'original_image':
        image = np.uint8(image * 255)