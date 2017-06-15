from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras.optimizers import RMSprop, SGD
from keras import backend as K
from keras.utils import to_categorical


def euclidean_distance(vects):
    x, y = vects
    return K.prod(K.stack([x, y],axis=1),axis=1)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1

def mean_acc(y_true, y_pred):
    a = [x[0] == y[0] for x,y in zip(y_true,y_pred)]
    return K.mean(a)

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
#    feature = Dropout(0.2, name='drop1')(feature)
    cls_out = Dense(751,activation='softmax', name='softmax')(feature)
    cls_model = Model(vgg16.input,cls_out)

    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    fea1,fea2 = base_model(input1), base_model(input2)
    cls1,cls2 = cls_model(input1), cls_model(input2)
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape,name='distance')([fea1, fea2])
    diff_out = Dense(2, activation='softmax', name='loss_diff')(distance)
    model = Model(inputs = [input1, input2], outputs = [diff_out,cls1,cls2])
    rms = RMSprop()
    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0005)
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],
            optimizer=sgd,loss_weights=[0.5,0.5,0.5],metrics=['accuracy'])
    return model
