from keras.applications import resnet50
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer as LB
from keras.preprocessing.image import ImageDataGenerator as IDG
from utils import image_quintuple_generator as iqg
from keras.layers import Dense,Input,Lambda
from keras.models import Model
import utils
import numpy as np
from keras import backend as  K


def image_triple_generator(lst_files,input_shape,batch_size,crop_shape=None):
    pos_ratio, neg_ratio = 1,1
    pos_limit, neg_limit = 1,4
    pos_factor, neg_factor = 1,1.01
    img_cache = {}
    datagen_args = dict(horizontal_flip = True)
    datagen_left = IDG(**datagen_args)
    datagen_right = IDG(**datagen_args)
    f = np.load(lst_files)
    lst,y = f['lst'],f['label']
    num_batches = len(y) // batch_size + 1
    clss = np.unique(y)
    num_clss = clss.shape[0]
    kmap = { v:k for k,v in enumerate(clss)}
    label_set = [np.where(y == c)[0] for c in clss]
    step = 0
    while True:
        step += 1
        #loop per epoch
        for bid in range(num_batches):
            id_left, id_right, y_diff = utils.gen_pairs(y,kmap,label_set, batch_size,pos_ratio, neg_ratio)
            Xleft = utils.process_images([lst[i] for i in id_left], datagen_left,
                    img_cache,input_shape,crop_shape)
            Xright = utils.process_images([lst[i] for i in id_right], datagen_right,
                    img_cache,input_shape,crop_shape)
            Y_diff = np.array(y_diff)
            yield [Xleft, Xright], Y_diff
            if step % 10 is 0:
                pos_ratio = min(pos_ratio * pos_factor, pos_limit)
                neg_ratio = min(neg_ratio * neg_factor, neg_limit)

def euclidean_distance(vects):
    x, y = vects
    return K.square(x- y)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1

input_shape = (256,256,3,)
crop_shape = (224,224,3,)
epochs = 10
steps_per_epoch = 800
base_model = 'resnet50'
batch_size = 16
lst_file = '../data/train.lst.npz'


resnet = resnet50.ResNet50(weights='imagenet')
feature_model = Model(resnet.input,resnet.layers[-2].output)
input1 = Input(shape=crop_shape,name='input1')
input2 = Input(shape=crop_shape,name='input2')
fea1 = feature_model(input1)
fea2 = feature_model(input2)
distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape,name='distance')([fea1, fea2])
diff_out = Dense(2, activation='softmax', name='y_diff')(distance)
model = Model(inputs = [input1, input2], outputs = diff_out)
sgd = SGD(lr=0.001,momentum=0.9,decay=0.0005)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
check_pointer = ModelCheckpoint(filepath='../models/indentify.weight.{epoch:02d}.hdf5',verbose=1,save_best_only=False,save_weights_only=True)
gen = image_triple_generator(lst_file,batch_size=batch_size,input_shape=input_shape, crop_shape=crop_shape)
model.fit_generator(gen,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[check_pointer])

del data
query_file = '../data/query.lst.npz'
test_file = '../data/test.lst.npz'
f  = np.load(query_file)
query_lst = f['lst']
query_data =  utils.extract_data_from_lst(query_lst,input_shape=crop_shape)
feature = feature_model.predict(query_data)
np.savez('../data/query_feature',feature=feature,label=f['label'],cam=f['cam'])

del query_data, data
f = np.load(test_file)
test_lst = f['lst']
test_data = utils.extract_data_from_lst(test_lst,input_shape=crop_shape)
feature = feature_model.predict(test_data)
np.savez('../data/test_feature',feature=feature,label=f['label'],cam=f['cam'])
