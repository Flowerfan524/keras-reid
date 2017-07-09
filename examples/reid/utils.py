import numpy as np
from PIL import Image
import os
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator as IDG
from sklearn.preprocessing import LabelBinarizer as LB


def extract_data_from_lst(lst,input_shape,crop_shape=None, preprocess=True):
    x = []
    for file in lst:
        im = read_input_img(file,input_shape,crop_shape)
        x += [np.asarray(im,dtype='float32')]
    x = np.array(x)
    if preprocess:
        x = img_process(x)
    return x

def crop_image(im,crop_shape):
    assert im.width > crop_shape[0] and im.height > crop_shape[1], 'error crop size'
    a = np.random.randint(im.width-crop_shape[0]+1)
    b = np.random.randint(im.height-crop_shape[1]+1)
    c = a + crop_shape[0]
    d = b + crop_shape[1]
    return im.crop((a,b,c,d))

def random_flip(im,seed=None):
    a = np.random.rand()
    if a > 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return im

def generate_train_lst(dire):
    x = []
    y = []
    cam = []
    #files = os.listdir(dire)
    files = sorted(os.listdir(dire))
    for file in files:
        if file.endswith('.jpg'):
            x += [dire + file]
            if file.startswith('-'):
                y += [-1]
                cam += [int(file[4])]
            else:
                y += [int(file[:4])]
                cam += [int(file[6])]

    return np.array(x),np.array(y),np.array(cam)

def read_input_img(file,shape,crop_shape=None):
    im = Image.open(file)
    im = random_flip(im)
    im = im.resize((shape[0],shape[1]))
    if crop_shape is None:
        return im
    im = crop_image(im,crop_shape=crop_shape)
    return im


def gen_pairs(s,bid,y, kmap, label_set, batch_size, pos_ratio, neg_ratio):
    id_left = np.random.randint(0,len(y),batch_size).tolist()
    #id_left = s[bid*batch_size:(bid+1)*batch_size]
    num_clss = len(label_set)
    id_right = []
    y_diff,y_cls1,y_cls2 = [],[],[]
    threshold = pos_ratio/(pos_ratio + neg_ratio)
    for idx in id_left:
        v = np.random.rand()
        ln = kmap[y[idx]]
        y_cls1 += [to_categorical(ln,num_clss).squeeze()]
        if v > threshold:
            ln = (np.random.randint(1,num_clss) + ln) % num_clss
            y_diff += [[0,1]]
        else:
            y_diff += [[1,0]]
        rn = np.random.randint(len(label_set[ln]))
        id_right += [label_set[ln][rn]]
        y_cls2 += [to_categorical(ln,num_clss).squeeze()]
    return id_left, id_right, y_diff



def image_quintuple_generator(lst_files,input_shape,batch_size,crop_shape=None):
    pos_ratio, neg_ratio = 1,1
    pos_limit, neg_limit = 1,4
    pos_factor, neg_factor = 1,1.01
    img_cache = {}
    f = np.load(lst_files)
    lst,y = f['lst'],f['label']
    num_ins = len(y)
    num_batches = num_ins // batch_size + 1
    clss = np.unique(y)
    num_clss = clss.shape[0]
    kmap = { v:k for k,v in enumerate(clss)}
    label_set = [np.where(y == c)[0] for c in clss]
    s = np.arange(num_ins)
    while True:
        s=np.random.permutation(s)
        #loop per epoch
        for bid in range(num_batches):
            id_left, id_right, y_diff = gen_pairs(s,bid,y,kmap,label_set, batch_size,pos_ratio, neg_ratio)
            Xleft = process_images([lst[i] for i in id_left],img_cache,input_shape,crop_shape)
            Xright = process_images([lst[i] for i in id_right],img_cache,input_shape,crop_shape)
            Y_diff = np.array(y_diff)
            Y_cls1 = np.array([to_categorical(kmap[y[i]],num_clss).squeeze() for i in id_left])
            Y_cls2 = np.array([to_categorical(kmap[y[i]],num_clss).squeeze() for i in id_right])
            yield [Xleft, Xright], [Y_diff, Y_cls1, Y_cls2]
            pos_ratio = min(pos_ratio * pos_factor, pos_limit)
            neg_ratio = min(neg_ratio * neg_factor, neg_limit)

def cache_read(img_name, img_cache,input_shape,crop_shape):
    if img_name not in img_cache:
        img = read_input_img(img_name,input_shape,crop_shape)
        img_cache[img_name] = img
    return img_cache[img_name]

def process_images(img_names, img_cache,input_shape,crop_shape):
    if crop_shape is None:
        X = np.zeros((len(img_names), input_shape[0], input_shape[1], 3))
    else:
        X = np.zeros((len(img_names), crop_shape[0], crop_shape[1], 3))
    for idx, img_name in enumerate(img_names):
        im = cache_read(img_name, img_cache,input_shape,crop_shape)
        X[idx] = np.asarray(im,dtype='float32')
    X[:,:,:,2] -= 97.8286
    X[:,:,:,1] -= 99.0468
    X[:,:,:,0] -= 105.606

    return X


def img_process(imgs, shift = (97.8286,99.046,105.606)):
    imgs[:,:,:,0] -= shift[2]
    imgs[:,:,:,1] -= shift[1]
    imgs[:,:,:,2] -= shift[0]
    #imgs[:,:,:,0] /= 255
    #imgs[:,:,:,1] /= 255
    #imgs[:,:,:,2] /= 255
    return imgs



def create_pairs(x,y,neg_times=1):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    neg_size = neg_times
    clss = np.unique(y)
    num_clss = len(clss)
    digit_indices = [np.where(y == c)[0] for c in clss]
    pairs = []
    label_diff = []
    label_clss = []
    for d in range(num_clss):
        n = len(digit_indices[d])
        for i in range(n):
            inc = np.random.randint(1,n)
            dn = (i + inc) % n
            z1, z2 = digit_indices[d][i], digit_indices[d][dn]
            l1 = to_categorical(d, num_clss).squeeze()
            pairs += [[x[z1], x[z2]]]
            label_diff += [[1,0]]
            label_clss += [[l1, l1]]
            incs = np.random.randint(1, num_clss, neg_size)
            dns = (incs + d) % num_clss
            dxs = [np.random.randint(len(digit_indices[dn])) for dn in dns]
            for idx1, idx2 in zip(dns, dxs):
                z1, z2 = digit_indices[d][i],digit_indices[idx1][idx2]
                l2 = to_categorical(idx1, num_clss).squeeze()
                pairs += [[x[z1], x[z2]]]
                label_diff += [[0,1]]
                label_clss += [[l1, l2]]
    return pairs, label_diff, label_clss



if __name__ == '__main__':
    train_dire = '/home/zqx/Desktop/flowerfan/data/Market-1501-v15.09.15/bounding_box_train/'
    test_dire = '/home/zqx/Desktop/flowerfan/data/Market-1501-v15.09.15/bounding_box_test/'
    query_dire = '/home/zqx/Desktop/flowerfan/data/Market-1501-v15.09.15/query/'


    x_train, y_train, cam_train = generate_train_lst(train_dire)
    x_test, y_test, cam_test = generate_train_lst(test_dire)
    x_query, y_query, cam_query = generate_train_lst(query_dire)

    np.savez('../data/train.lst', lst=x_train, label=y_train, cam=cam_train)
    np.savez('../data/test.lst', lst=x_test, label=y_test, cam=cam_test)
    np.savez('../data/query.lst', lst=x_query, label=y_query, cam=cam_query)

    lst_pairs, y_diff, y_clss = create_pairs(x_train, y_train)
    tuples = np.array([(x[0],x[1],y,z[0],z[1]) for x,y,z in zip(lst_pairs, y_diff, y_clss)], dtype=object)
    np.savez('../data/input.lst', tuples = tuples)
