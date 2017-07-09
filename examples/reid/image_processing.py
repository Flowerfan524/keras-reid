import numpy as np
from PIL import Image 


def random_crop(im,crop_shape,seed=None):
    assert crop_shape[0] < im.width and crop_shape[1] < im.height, 'wrong crop shape'
    np.random.seed(seed)
    a = np.random.randint(im.width - crop_shape[0] + 1)
    b = np.random.randint(im.height - crop_shape[1] + 1)
    c = a + crop_shape[0]
    d = b + crop_shape[1]
    im = im.crop((a,b,c,d))
    return im

def random_flip(im,seed=None):
    np.random.seed(seed)
    a  = np.random.rand()
    if a > 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return im

def img_to_array(im):
    return np.asarray(im,dtype='float32')

def load_img(file):
    im = Image.open(file)
    return im

def resize_img(im,target_shape):
    im = im.resize(target_shape)
    return im

def get_img(file,resize_shape=None,crop_shape=None,seed=None):
    im = load_img(file)
    im = random_flip(im,seed)
    im = resize_img(im,resize_shape)
    im = random_crop(im,crop_shape,seed)
    return img_to_array(im)
