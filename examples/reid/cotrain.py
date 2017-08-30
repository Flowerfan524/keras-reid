import reid_net
import numpy as np
import copy
import utils
from utils import image_quintuple_generator as iqg
from keras.models import Model
from sklearn.preprocessing import LabelBinarizer as LB
from keras.optimizers import SGD
from utils import image_base_generator as ibg
import model



def sel_idx(score, y, ratio=0.1):
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    kmap = {v:k for k,v in enumerate(clss)}
    y_ture = np.array([kmap[i] for i in y])
    count_per_class = [sum(y==c) for c in clss]
    pred_y = np.argmax(score,axis=1)
    for cls in range(score.shape[1]):
        indices = np.where(pred_y == cls)[0]
        cls_score = score[indices,cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(count_per_class[cls] * ratio)), indices.shape[0])
        add_indices[indices[idx_sort[:add_num]]] = 1

    return add_indices.astype('bool')




def cotrain(train_lst_file,untrain_lst_file,mname1,mname2,params1,params2):
    train_data = np.load(train_lst_file)
    untrain_data = np.load(untrain_lst_file)
    model1 = model.get_model(model_name=mname1)
    model2 = model.get_model(model_name=mname2)
    optimizer1=SGD(lr=0.01,momentum=0.9,decay=0.005)
    optimizer2=SGD(lr=0.001,momentum=0.9,decay=0.005)
    model.train_model(model1,train_data,optimizer1,params1)
    model.train_model(model2,train_data,optimizer2,params2)
    clss = np.unique(train_data['label'])
    kmap = {v:k for k,v in enumerate(clss)}
    X_untrain1 = utils.extract_data_from_lst(untrain_data['lst'],params1['crop_shape'])
    X_untrain2 = utils.extract_data_from_lst(untrain_data['lst'],params2['crop_shape'])
    lst_untrain = untrain_data['lst'] 
    lst1,lst2 = copy.deepcopy(train_data['lst']),copy.deepcopy(train_data['lst'])
    lst1,lst2 = list(lst1),list(lst2)
    y_train1,y_train2 = copy.deepcopy(train_data['label']),copy.deepcopy(train_data['label'])
    for step in range(5):

        # select unlabel data# feature_model1 = Model()
        # feature_model2 = Model()
        print('select unlabel data')
        prob1 = model1.predict(X_untrain1)
        prob2 = model2.predict(X_untrain2)
        pred_y = np.argmax(prob1 + prob2, axis=1)
        add_idx1 = sel_idx(prob2, train_data['label'])
        add_idx2 = sel_idx(prob1, train_data['label'])

        #add unlabel data and train
        print('train with augmented data')
        lst1.extend(list(lst_untrain[add_idx1]))
        lst2.extend(list(lst_untrain[add_idx2]))
        add_y1 = clss[pred_y[add_idx1]]
        add_y2 = clss[pred_y[add_idx2]]
        y_train1 = np.hstack((y_train1,add_y1))
        y_train2 = np.hstack((y_train2,add_y2))

        train1 = {'lst':lst1, 'label':y_train1}
        train2 = {'lst':lst2, 'label':y_train2}

        model1 = model.get_model(model_name=mname1)
        model2 = model.get_model(model_name=mname2)
        model.train_model(model1,train1,optimizer1,params1)
        model.train_model(model2,train2,optimizer2,params2)

        # remove add untrain lst
        print('remove used data')
        add_idx = add_idx1 + add_idx2
        lst_untrain = lst_untrain[~add_idx]
        X_untrain1 = X_untrain1[~add_idx]
        X_untrain2 = X_untrain2[~add_idx]
        print('Add train size view 1: {}, view 2:{}'
              .format(add_y1.shape[0], add_y2.shape[0]))
    return model1,model2



def spaco(train_lst,untrain_lst,mname1,mname2,params):
    train_data = np.load(train_lst)
    untrain_data = np.load(untrain_lst)
    clss = np.unique(train_data['label'])
    kmap = {v:k for k,v in enumerate(clss)}
    X_untrain = utils.extract_data_from_lst(untrain_data['lst'])
    model1 = model.get_model(model_name=mname1)
    model2 = model.get_model(model_name=mname2)
    optimizer=SGD(lr=0.001,momentum=0.9,decay=0.005)
    model.train_model(model1,train_data,optimizer,params)
    model.train_model(model2,train_data,optimizer,params)
    y_train1,y_train2 = copy.deepcopy(train_data['label']),cppy.deepcopy(train_data['label'])
    prob1 = model1.predict(X_untrain)
    prob2 = model2.predict(X_untrain)
    pred_y = np.argmax(prob1 + prob2, axis=1)
    add_ratio = 1
    gamma = 1
    for step in range(5):

        # select unlabel data from model1
        idx_conf = sel_idx(prob1, train_data['label'])
        idx_cls = np.argmax(prob1[idx_conf,:], axis=1)

        # add data to model 2
        score_conf = pred_prob2
        score_conf[idx_conf,idx_cls] += gamma
        add_idx = sel_idx(score_conf, train_data['label'], add_ratio=1)
        add_y = clss(kmap[pred_y[add_idx]])


        #add unlabel data and train
        lst = copy.deepcopy(train_data['lst'])
        lst.extend(lst_untrain[add_idx])
        y_train = np.vstack((train_data['label'],add_y))
        train = {'lst':lst, 'label':y_train}
        model.train_model(model2,train,optimizer,params)


        print('Add train size view 2: {}'
              .format(add_y.shape[0]))

        # tune label
        prob2 = model2.predict(X_untrain)
        pred_y = np.argmax(prob1 + prob2, axis=1)


        add_ratio += 0.5
        # select unlabel data from model2
        idx_conf = sel_idx(prob2, train_data['label'])
        idx_cls = np.argmax(prob2[idx_conf,:], axis=1)


        # add data to model 1
        score_conf = pred_prob1
        score_conf[idx_conf,idx_cls] += gamma
        add_idx = sel_idx(score_conf, train_data['label'],add_ratio)
        add_y = clss(kmap[pred_y[add_idx]])

        #add unlabel data and train
        lst = copy.deepcopy(train_data['lst'])
        lst.extend(lst_untrain[add_idx])
        y_train = np.vstack((train_data['label'],add_y))
        train = {'lst':lst, 'label':y_train}
        model.train_model(model1,train,optimizer,params)

        # tune label
        prob1 = model1.predict(X_untrain)
        pred_y = np.argmax(prob1 + prob2, axis=1)

        add_ratio += 0.5
        print('Add train size view 1: {}'
              .format(add_y.shape[0]))
    return model1,model2
