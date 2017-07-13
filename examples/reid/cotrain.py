import reid_net
import numpy as np
import copy
import utils
from utils import image_quintuple_generator as iqg
from keras.models import Model
from sklearn.preprocessing import LabelBinarizer as LB

def get_data(file):
    f = np.load(file)
    return f['data'],f['label']


def sel_idx(score, ratio):
    sel_idx = []
    for cls in range(score.shape[1]):


    return None

def cotrain(lst_train_name,unlst_train_name,mname1,mname2):
    input_shape =
    lst_train, y_train = get_data(lst_train_name)
    lst_untrain ,y_untrain = get_data(unlst_train_name)
    clss = np.unique(y_train)
    X_untrain = utils.extract_data_from_lst(lst_untrain)
    tuples_train = utils.create_pairs(lst_train,y_train)
    model1 = reid_net.reid_net(base_model=mname1,input_shape=input_shape)
    model2 = reid_net.reid_net(base_model=mname2,input_shape=input_shape)
    model1.fit_generator(iqg(tuples_train,batch_size=batch_size),
                         steps_per_epoch=3000,epoch=6)
    model2.fit_generator(iqg(tuples_train,batch_size=batch_size),
                         steps_per_epoch=3000,epoch=6)
    lst1,lst2 = copy.deepcopy(lst_train),cooy.deepcopy(lst_train)
    y_train1,y_train2 = copy.deepcop(y_train),cppy.deepcopy(y_train)
    for step in range(5):

        # select unlabel data
        prob_model1 = Model(inputs=model1.input[0],outputs=model1.output[1])
        prob_model2 = Model(inputs=model2.input[0],outputs=model2.output[1])
        # feature_model1 = Model()
        # feature_model2 = Model()
        prob1 = prob_model1.predict(X_untrain)
        prob2 = prob_model2.predict(X_untrain)
        pred_y = np.argmax(prob1 + prob2, axis=1)
        add_idx1 = sel_idx(prob2, add_ratio)
        add_idx2 = sel_idx(prob1, add_ratio)

        #add unlabel data and train
        lst1.extend(lst_untrain[add_idx1])
        lst2.extend(lst_untrain[add_idx2])
        add_y1 = clss(pred_y[add_idx1])
        add_y2 = clss(pred_y[add_idx2])
        y_train1 = np.vstack((y_train1,add_y1))
        y_train2 = np.vstack((y_train2,add_y2))
        tuples1 = utils.create_pairs(lst1,y_train1)
        tuples2 = utlis.create_pairs(lst2,y_train2)
        model1 = reid_net.reid_net(base_model=mname1,input_shape=input_shape)
        model2 = reid_net.reid_net(base_model=mname2,input_shape=input_shape)
        model1.fit_generator(iqg(tuples=tuples1,batch_size=batch_size),
                             steps_per_epoch=3000,epoch=6)
        model2.fit_generator(iqg(tuples=tuples2,batch_size=batch_size),
                             steps_per_epoch=3000,epoch=6)

        # remove add untrain lst
        add_idx = add_idx1 + add_idx2
        lst_untrain = lst_untrain[~add_idx]
        y_untrain = y_untrain[~add_idx]
        print('Add train size view 1: {}, view 2:{}'
              .format(add_y1.shape[0], add_y2.shape[0]))
    return model1,model2
