import numpy as np
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator as IDG


def extract_data_from_lst(lst, preprocess=True):
    x = []
    for file in lst:
        x += [read_input_img(file)]
    x = np.array(x)
    if preprocess:
        x = img_process(x)
    return x


def generate_train_lst(dire):
    x = []
    y = []
    cam = []
    for file in os.listdir(dire):
        if file.endswith('.jpg'):
            x += [dire + file]
            if file.startswith('-'):
                y += [-1]
                cam += [int(file[4])]
            else:
                y += [int(file[:4])]
                cam += [int(file[6])]

    return np.array(x),np.array(y),np.array(cam)

def read_input_img(file):
    im = image.load_img(file, target_size=(224,224,))
    im = image.img_to_array(im)
    return im

def image_quintuple_generator(img_quintuples, batch_size):
    datagen_args = dict(width_shift_range = 0.1,
                        height_shift_range= 0.1,
                        horizontal_flip = True)
    datagen_left = IDG(**datagen_args)
    datagen_right = IDG(**datagen_args)
    img_cache = {}

    while True:
        #loop per epoch
        num_recs = len(img_quintuples)
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size:(bid + 1) * batch_size]
            batch = [img_quintuples[i] for i in batch_indices]
            seed = np.random.randint(0,100,1)[0]
            Xleft = process_images([b[0] for b in batch], seed, datagen_left, img_cache)
            Xright = process_images([b[1] for b in batch], seed, datagen_right, img_cache)
            Y_diff = np.array([b[2] for b in batch])
            Y_cls1 = np.array([b[3] for b in batch])
            Y_cls2 = np.array([b[4] for b in batch])
            yield [Xleft, Xright], [Y_diff, Y_cls1, Y_cls2]

def cache_read(img_name, img_cache):
    if img_name not in img_cache:
        img = read_input_img(img_name)
        img_cache[img_name] = img
    return img_cache[img_name]

def process_images(img_names, seed, datagen, img_cache):
    np.random.seed(seed)
    X = np.zeros((len(img_names), 224, 224, 3))
    for idx, img_name in enumerate(img_names):
        img = cache_read(img_name, img_cache)
        X[idx] = datagen.random_transform(img)
    X[:,:,:,0] -= 97.8286
    X[:,:,:,1] -= 99.0468
    X[:,:,:,2] -= 105.606
    return X


def img_process(imgs, shift = (97.8286,99.0468,105.606)):
    imgs[:,:,:,0] -= shift[0]
    imgs[:,:,:,1] -= shift[1]
    imgs[:,:,:,2] -= shift[2]
   # imgs[:,:,:,0] /= 255
   # imgs[:,:,:,1] /= 255
   # imgs[:,:,:,2] /= 255
    return imgs


def draw_img(subplot, img, title):
    plt.subplot(subplot)
    plt.imshow(img)
    plt.title(title)
    plt.xsticks([])
    plt.ysticks([])



def create_pairs(x,y):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    neg_size = 3
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
            label_diff += [[0,1]]
            label_clss += [[l1, l1]]
            incs = np.random.randint(1, num_clss, neg_size)
            dns = (incs + d) % num_clss
            dxs = [np.random.randint(len(digit_indices[dn])) for dn in dns]
            for idx1, idx2 in zip(dns, dxs):
                z1, z2 = digit_indices[d][i],digit_indices[idx1][idx2]
                l2 = to_categorical(idx1, num_clss).squeeze()
                pairs += [[x[z1], x[z2]]]
                label_diff += [[1,0]]
                label_clss += [[l1, l2]]
    return pairs, label_diff, label_clss



if __name__ == '__main__':
    train_dire = '/home/zqx/Desktop/flowerfan/data/Market-1501-v15.09.15/bounding_box_train/'
    test_dire = '/home/zqx/Desktop/flowerfan/data/Market-1501-v15.09.15/bounding_box_test/'
    query_dire = '/home/zqx/Desktop/flowerfan/data/Market-1501-v15.09.15/query/'


    x_train, y_train, cam_train = generate_train_lst(train_dire)
    x_test, y_test, cam_test = generate_train_lst(test_dire)
    x_query, y_query, cam_query = generate_train_lst(query_dire)

    np.savez('../data/train.lst', data=x_train, label=y_train, cam=cam_train)
    np.savez('../data/test.lst', data=x_test, label=y_test, cam=cam_test)
    np.savez('../data/query.lst', data=x_query, label=y_query, cam=cam_query)

    lst_pairs, y_diff, y_clss = create_pairs(x_train, y_train)
    tuples = np.array([(x[0],x[1],y,z[0],z[1]) for x,y,z in zip(lst_pairs, y_diff, y_clss)], dtype=object)
    np.savez('../data/input.lst', tuples = tuples)
