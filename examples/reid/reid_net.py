from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical


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

def reid_net(include_top = True, input_shape = None):
    vgg16 = VGG16(include_top=False,input_shape=input_shape)
#    for layer in vgg16.layers:
#        layer.trainable = False
    x = vgg16.output
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation = 'relu', name='fc1')(x)
    feature = Dense(1024, activation = 'relu', name='fc2')(x)
    base_model = Model(vgg16.input,feature)
#    feature = Dropout(0.2,name='drop2')(x)
#    if not include_top:
#        return base_model
    feature = Dropout(0.2, name='drop1')(feature)
    cls_out = Dense(751,activation='softmax', name='softmax')(feature)
    cls_model = Model(vgg16.input,cls_out)

    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    fea1,fea2 = base_model(input1), base_model(input2)
    cls1,cls2 = cls_model(input1), cls_model(input2)
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([fea1, fea2])
    model = Model(inputs = [input1, input2], outputs = [distance,cls1,cls2])
    rms = RMSprop()
    model.compile(loss=['binary_crossentropy','categorical_crossentropy','categorical_crossentropy'],optimizer=rms,loss_weights=[0.5,0.5,0.5])
    return model


