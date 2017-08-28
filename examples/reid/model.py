from keras.applications import vgg16, xception, resnet50
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD
from utils import image_base_generator as ibg


def get_model(model_name,input_shape=(224,224,3)):
    if model_name is 'vgg16':
        base_model = vgg16.VGG16()
    elif model_name is 'xception':
        base_model = xception.Xception()
        input_shape = (299,299,3)
    elif model_name is 'resnet50':
        base_model = resnet50.resnet50()

    fea_model = Model(base_model.input, base_model.layers[-1].input)
    input1 = Input(shape=input_shape,name='input1')
    fea1 = fea_model(input1)
    cls1 = Dense(751, activation='softmax', name='class')
    model = Model(input1,cls1)
    return model

def train_model(model,data,optimizer,params):
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    gen = ibg(data,params.batch_size,prams.input_shape,params.crop_shape)
    model.fit_generator(gen,steps_per_epoch=params.steps_per_epoch,
                        epochs=params.epochs)
