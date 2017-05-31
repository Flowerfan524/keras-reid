from PIL import Image
import numpy as np
import os

train_dire = '/Users/flowerfan/Desktop/Market-1501-v15.09.15/bounding_box_train/'
test_dire = '/Users/flowerfan/Desktop/Market-1501-v15.09.15/bounding_box_test/'
query_dire = '/Users/flowerfan/Desktop/Market-1501-v15.09.15/query/'




def extract_imgs(dire):
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

    return x,y,cam

def read_input_img(file):
    im = Image.open(file)
    im = im.resize((256,256))
    return im


x_train, y_train, cam_train = extract_imgs(train_dire)
x_test, y_test, cam_test = extract_imgs(test_dire)
x_query, y_query, cam_query = extract_imgs(query_dire)

np.savez('market1501.npz', x_train=x_train, y_train=y_train, cam_train=cam_train
                         , x_test=x_test, y_test=y_test, cam_test=cam_test
                         , x_query=x_query, y_query=y_query, cam_query=cam_query)
