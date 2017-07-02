import numpy as np
from PIL import Image pil_image


def random_crop(im,crop_shape,seed=None):
    assert crop_shape[0] > im.width and crop_shape[1] > im.height, 'wrong crop shape'
    np.random.seed(seed)
    a = np.random.randint(im.width - crop_shape[0] + 1)
    b = np.random.randint(im.heigth - crop_shape[1] + 1)
    c = a + crop_shape[0]
    d = b + crop_shape[1]
    im = im.crop(（a,b,c,d))
    return im

def random_flip(im,seed=None):
    np.random.seed(seed)
    a  = np.random.rand()
    if a > 0.5:
        im = im.rotate(180)
    return im

def img_to_array(im):
    return np.asarray(im,dtype='float32')

def load_img(file):
    im = Image.open(file)
    retrun im

def reisze_img(im,target_shape):
    im = im.resize(target_shape)
    return im
