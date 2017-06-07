import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical
import random
import utils
import reid_net
from sklearn.preprocessing import LabelBinarizer as LB
from keras.image import ImageDataGenerator as IDG

input_shape = (224,224,3,)
epochs = 20

def create_pairs(x, y):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    clss = np.unique(y)
    num_clss = len(clss)
    digit_indices = [np.where(y == c)[0] for c in clss]
    pairs = []
    labels = []
    clf_labels1,clf_labels2 = [], []
    for d in range(num_clss):
        n = len(digit_indices[d])
        for i in range(n):
            inc = random.randrange(1,n)
            dn = (i + inc) % n
            z1, z2 = digit_indices[d][i], digit_indices[d][dn]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_clss)
            dn = (d + inc) % num_clss
            dx = random.randrange(len(digit_indices[dn]))
            z1, z2 = digit_indices[d][i], digit_indices[dn][dx]
            pairs += [[x[z1], x[z2]]]
            l1,l2 = to_categorical(d,len(clss)).squeeze(),to_categorical(dn,len(clss)).squeeze()
            labels += [1, 0]
            clf_labels1 += [l1,l1]
            clf_labels2 += [l1,l2]
    return np.array(pairs), np.array(labels), np.array(clf_labels1), np.array(clf_labels2)

f = np.load('../data/train.lst.npz')
x_lst, y_train, cam_train = f['data'], f['label'], f['cam']
x_train = utils.extract_data_from_lst(x_lst)
s = np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train, y_train, cam_train = x_train[s], y_train[s],cam_train[s]
tr_pairs, tr_y, tr_c1, tr_c2 = create_pairs(x_lst, y_train)

model = reid_net.reid_net(input_shape=input_shape)

rms = RMSprop()

data_gen = IDG(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True)
data_gen.fit(x_train)
batch_lst1 = np.array_split(tr_pairs[:,0], num_batch)
batch_lst2 = np.array_split(tr_pairs[:,1], num_batch)
batch_y = np.array_split(tr_y, num_batch)
batch_c1 = np.array_split(tr_c1, num_batch)
batch_c2 = np.array_split(tr_c2, num_batch)
assert len(batch_lst1) == len(batch_lst2), 'lst dismatch'
for epoch in range(epochs):
    batch_size = 16
    num_batch = int(tr_pairs.shape[0]/batch_size)
    for i in range(num_batch):
        input_data1 = utils.extract_data_from_lst(batch_lst1[i])
        input_data2 = utils.extract_data_from_lst(batch_lst2[i])
        for 
        model.train_on_batch([input_data1, input_data2], [batch_y[i], batch_c1[i], batch_c2[i]])
#        model.evaluate([input_data1, input_data2],[batch_y[i], batch_c1[i], batch_c2[i]])

model.save_weights('../data/model_weights.h5')


pred_model = Model(inputs=model.input[0], outputs=model.output[1])
x_train = utils.extract_data_from_lst(x_lst)
pred_y = pred_model.predict(x_train)
pred_y = np.argmax(pred_y, axis = 1)
lb =  LB()
lb.fit(y_train)
accuracy = np.mean(lb.classes_[pred_y] == y_train)
print('trained model accuracy {}'.format(accuracy))
