import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from utils import image_quintuple_generator as iqg
import reid_net
import utils
from sklearn.preprocessing import LabelBinarizer as LB


input_shape = (256,256,3,)
crop_shape = (224,224,3,)
epochs = 5
steps_per_epoch = 4000
base_model = 'resnet50'
batch_size = 16
lst_file = '../data/train.lst.npz'

model = reid_net.reid_net(input_shape=crop_shape,base_model=base_model)
#model.load_weights('../models/resnet50.weight.03.hdf5')
check_pointer = ModelCheckpoint(filepath='../models/'+base_model+'.weight.{epoch:02d}.hdf5',
        verbose=1,save_best_only=False,save_weights_only=True)

model.fit_generator(iqg(lst_file,batch_size=batch_size,
        input_shape=input_shape,crop_shape=crop_shape)
            ,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[check_pointer])

model.save_weights('../models/model_weights.h5')


