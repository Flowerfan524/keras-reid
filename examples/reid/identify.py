from keras.applications import resnet50, vgg16,inception_v3
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer as LB
from keras.utils import to_categorical
from keras.layers import Dense,Input,Dropout
from keras.models import Model
import utils
import numpy as np
import image_processing as ipo

model = 'resnet50'
batch_size = 32
epochs = 15
steps_per_epoch = 1500
input_shape = (256,256,3)
crop_shape = (224,224,3)

if model == 'inception':
    input_shape = (321,321,3)
    crop_shape = (299,299,3)
    base_model = inception_v3.InceptionV3(weights='imagenet')
elif model == 'resnet50':
    base_model = resnet50.ResNet50(weights='imagenet',include_top=False,pooling='avg')
elif model == 'vgg16':
    base_model = vgg16.VGG16(weights='imagenet')
else:
    print('wrong model')
train_lst_file = '../data/train.lst.npz'
f = np.load(train_lst_file)
train_lst, train_y = f['lst'], f['label']


def read_img(lst_file,input_shape,crop_shape):
    X = np.zeros((len(lst_file),)+crop_shape)
    for idx,lst in enumerate(lst_file):
        X[idx] = ipo.get_img(lst,(input_shape[0],input_shape[1]),(crop_shape[0],crop_shape[1]))
    X[:,:,:,0] -= 105.606
    X[:,:,:,1] -= 99.0468
    X[:,:,:,2] -= 97.8286
    return X

def gen_data(data,batch_size,input_shape,crop_shape=None):
    lst,y = data['lst'],data['label']
    num_ins = len(y)
    clss = np.unique(y)
    num_clss = clss.shape[0]
    num_batchs = num_ins // batch_size
    kmap = {v:k for k,v in enumerate(clss)}
    s = np.arange(num_ins)
    while True:
        s = np.random.permutation(s)
        for batch in range(num_batchs):
            indices = s[batch*batch_size:(batch+1)*batch_size]
            X = read_img(lst[indices],input_shape,crop_shape)
            label = np.array([to_categorical(kmap[y[i]],num_clss).squeeze() for i in indices])
            yield X,label

base_model = resnet50.ResNet50(weights='imagenet',include_top=True)
feature_model = Model(base_model.input,base_model.layers[-1].input)
cls_out = Dense(751,activation='softmax',name='y_clss')
input1 = Input(shape=crop_shape,name='input1')
fea1 = feature_model(input1)
#drop1 = Dropout(0.1,name='drop1')(fea1)
cls1 = cls_out(fea1)
model = Model(inputs=input1,outputs=cls1)
#sgd = SGD(lr=0.001,momentum=0.9,decay=0.0005)
sgd = SGD()
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
check_pointer = ModelCheckpoint(filepath='../models/identify.weight.{epoch:02d}.hdf5',verbose=1,save_best_only=False,save_weights_only=True)
lb = LB()
lb.fit(train_y)
y = lb.transform(train_y)
data = utils.extract_data_from_lst(train_lst,input_shape,crop_shape)
#model.load_weights('../models/identify.weight.19.hdf5')
model.fit(x=data,y=y,epochs=epochs,batch_size=batch_size,callbacks=[check_pointer])
#gen = gen_data(train_lst_file,batch_size,input_shape,crop_shape)
#model.fit_generator(gen,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[check_pointer])


#feature_model = Model(model.input,model.layers[-1].input)
query_file = '../data/query.lst.npz'
test_file = '../data/test.lst.npz'
f  = np.load(query_file)
query_lst = f['lst']
query_data =  utils.extract_data_from_lst(query_lst,input_shape=crop_shape)
feature = feature_model.predict(query_data)
np.savez('../data/query_feature',feature=feature,label=f['label'],cam=f['cam'])
del query_data, feature
f = np.load(test_file)
test_lst = f['lst']
test_data = utils.extract_data_from_lst(test_lst,input_shape=crop_shape)
feature = feature_model.predict(test_data)
np.savez('../data/test_feature',feature=feature,label=f['label'],cam=f['cam'])
