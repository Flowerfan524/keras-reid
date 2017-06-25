from keras.applications import vgg16, inception_v3, resnet50, xception
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras import backend as K
from keras.utils import to_categorical


def euclidean_distance(vects):
    x, y = vects
    return K.square(x- y)

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

def reid_net(base_model='vgg16',include_top = True, input_shape = None):
    if base_model is 'vgg16':
        vgg = vgg16.VGG16(include_top=True,weights='imagenet',input_shape=input_shape)
        #x = vgg.output
        #x = Flatten(name='flatten')(x)
        #x = Dense(1024, activation = 'relu', name='fc1')(x)
        #x = Dense(1024, activation = 'relu', name='fc2')(x)
        base_model = Model(vgg.input,vgg.get_layer('fc2').output)
    elif base_model is 'inception':
        inception = inception_v3.InceptionV3(include_top=True, weights='imagenet',input_shape=input_shape)
        base_model = Model(inputs=inception.input, outputs=inception.get_layer('avg_pool').output)
    elif base_model is 'resnet50':
        resnet = resnet50.ResNet50(include_top=True, weights='imagenet',input_shape=input_shape)
        base_model = Model(inputs=resnet.input, outputs=resnet.layers[-2].output)

    if not include_top:
        return base_model
    feature = base_model.output
    cls_out = Dense(751,activation='softmax', name='softmax')(feature)
    cls_model = Model(base_model.input,cls_out)
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    fea1,fea2 = base_model(input1), base_model(input2)
    cls1,cls2 = cls_model(input1), cls_model(input2)
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape,name='distance')([fea1, fea2])
    diff_out = Dense(2, activation='softmax', name='loss_diff')(distance)
    model = Model(inputs = [input1, input2], outputs = [diff_out,cls1,cls2])
    return model
