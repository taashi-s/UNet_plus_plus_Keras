import os
import glob
import numpy as np
from PIL import Image
import cv2


def load_images(dir_name, image_shape, with_normalize=True):
    files = glob.glob(os.path.join(dir_name, '*.png'))
    files.sort()
    h, w, ch = image_shape
    images = []
    print ('load_images : ', len(files))
    for i, file in enumerate(files):
        img = load_image(file, image_shape, with_normalize=with_normalize)
        images.append(img)
        if i % 500 == 0:
            print('load_images loaded ', i)
    return (files, np.array(images, dtype=np.float32))


def load_image(file_name, image_shape, with_normalize=True):
    src_img = cv2.imread(file_name)
    if src_img is None:
        return None

    dist_img = src_img
    if image_shape[2] == 1:
        dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2GRAY)
    dist_img = cv2.resize(dist_img, (image_shape[0], image_shape[1]))
    if with_normalize:
        dist_img = dist_img / 255
    return dist_img


def save_images(dir_name, image_data_list, file_name_list):
    for _, (image_data, file_name) in enumerate(zip(image_data_list, file_name_list)):
        name = os.path.basename(file_name)
        save_path = os.path.join(dir_name, name)
        cv2.imwrite(save_path, image_data * 255)
        print('saved : ' , save_path)
