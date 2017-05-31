from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical
import random
import utils


input_shape = (224,224,3,)
epochs = 28

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, y):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    clss = np.unique(y)
    num_clss = len(clss)
    digit_indices = [np.where(y == c) for c in clss]
    pairs = []
    labels = []
    clf_labels1,clf_labels2 = [], []
    n = min([len(digit_indices[d]) for d in range(num_clss)]) - 1
    for d in range(num_clss):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_clss)
            dn = (d + inc) % num_clss
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            l1,l2 = to_categorical(d,len(clss)).squeeze(),to_categorical(dn,len(clss)).squeeze()
            labels += [1, 0]
            clf_labels1 += [l1,l1]
            clf_labels2 += [l1,l2]
    return np.array(pairs), np.array(labels), np.array(clf_labels1), np.array(clf_labels2)

f = np.load('market1501.npz')
x_lst, y_train, cam_train = f['x_train'], f['y_train'], f['cam_train']
x_train = []
for file in x_lst:
    im = utils.read_input_img(file)
    x_train += [np.array(im)]

tr_pairs, tr_y, tr_c1, tr_c2 = create_pairs(x_train, y_train)

vgg16 = VGG16(weights='imagenet', include_top=False)

x = vgg16.output
x = Flatten(name='flatten')(x)
x = Dense(4096, activation = 'relu', name='fc1')(x)
x = Dropout(0.2,name='drop1')(x)
x = Dense(4096, activation = 'relu', name='fc2')(x)
feature = Dropout(0.2,name='drop2')(x)
base_model = Model(vgg16.input,feature)
cls_out = Dense(751,activation='softmax', name='softmax')(base_model.output)
cls_model = Model(vgg16.input,cls_out)
input1 = Input(shape=input_shape)
input2 = Input(shape=input_shape)
fea1,fea2 = base_model(input1), base_model(input2)
cls1,cls2 = cls_model(input1), cls_model(input2)
distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([fea1, fea2])

model = Model(inputs = [input1, input2], outputs = [distance,cls1,cls2])

# train
rms = RMSprop()
model.compile(loss=[contrastive_loss,'categorical_crossentropy','categorical_crossentropy'],
              optimizer=rms,
              loss_weights=[1.,0.5,0.5])

model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], [tr_y,tr_c1,tr_c2],
          batch_size=128,
          epochs=epochs)

