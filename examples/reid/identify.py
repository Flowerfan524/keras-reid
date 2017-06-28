from keras.applications import resnet50
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer as LB
from keras.layers import Dense,Input
from keras.models import Model
import utils
import numpy as np

f = np.load('../data/train.lst.npz')
train_lst, train_y = f['lst'],f['label']
input_shape = (256,256,3)
crop_shape = (224,224,3)
batch_size = 16

resnet = resnet50.ResNet50(weights='imagenet')
feature_model = Model(resnet.input,resnet.layers[-2].output)
cls_out = Dense(751,activation='softmax',name='y_clss')
input1 = Input(shape=crop_shape,name='input1') 
fea1 = feature_model(input1)
cls1 = cls_out(fea1)
model = Model(inputs=input1,outputs=cls1)
sgd = SGD(lr=0.001,momentum=0.9,decay=0.0005)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
check_pointer = ModelCheckpoint(filepath='../models/indentify.weight.{epoch:02d}.hdf5',verbose=1,save_best_only=False,save_weights_only=True)
lb = LB()
lb.fit(train_y)
y = lb.transform(train_y)
data = utils.extract_data_from_lst(train_lst,input_shape,crop_shape)
model.fit(x=data,y=y,epochs=10,batch_size=batch_size,callbacks=[check_pointer])

del data
query_file = '../data/query.lst.npz'
test_file = '../data/test.lst.npz'
f  = np.load(query_file)
query_lst = f['lst']
query_data =  utils.extract_data_from_lst(query_lst,input_shape=crop_shape)
feature = feature_model.predict(query_data)
np.savez('../data/query_feature',feature=feature,label=f['label'],cam=data['cam'])

del query_data, data
f = np.load(test_lst)
test_lst = f['lst']
test_data = utils.extract_data_from_lst(test_lst,input_shape=crop_shape)
feature = feature_model.predict(test_data)
np.savez('../data/test_feature',feature=feature,label=f['label'],cam=data['cam'])



