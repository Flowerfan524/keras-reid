import numpy as np
import cotrain
import numpy as np
import utils
from scipy.spatial.distance import cdist
import evaluation



name_model1 = 'xception'
name_model2 = 'resnet50'

params1 = {
    'input_shape': (336,336,3),
    'crop_shape': (299,299,3),
    'epochs': 40,
    'steps_per_epoch': 250,
    'batch_size': 16,
}

params2 = {
    'input_shape': (256,256,3),
    'crop_shape': (224,224,3),
    'epochs': 40,
    'steps_per_epoch': 250,
    'batch_size': 16,
}

train_lst = '../data/train.lst.npz'

def split_data(train_lst, ratio=0.2):
    f = np.load(train_lst)
    lst, y = f['lst'], f['label']
    lst_train = []
    y_train = []
    lst_untrain = []
    y_untrain = []
    clss = np.unique(y)
    for cls in clss:
        indices = np.where(y == cls)[0]
        np.random.shuffle(indices)
        mid_idx = int(np.ceil(indices.shape[0] * ratio))
        lst_train += list(lst[indices[:mid_idx]])
        lst_untrain += list(lst[indices[mid_idx:]])
        y_train += list(y[indices[:mid_idx]])
        y_untrain += list(y[indices[mid_idx:]])
    np.savez('../data/train_lst', lst=lst_train, label=y_train)
    np.savez('../data/untrain_lst', lst=lst_untrain, label=y_untrain)

                                
#split_data(train_lst)
train_lst = '../data/train_lst.npz'
untrain_lst = '../data/untrain_lst.npz'

model1,model2 = cotrain.cotrain(train_lst,untrain_lst,name_model1,name_model2,params1,params2)
model1.save_weights('cotrain_xception.h5')
model2.save_weights('cotrain_resnet.h5')

fea_mod1 = Model(model1.input,model1.layers[-1].input)
fea_mod2 = Model(model2.input,model2.layers[-2].input)

query_file = '../data/query.lst.npz'
test_file  = '../data/test.lst.npz'
f = np.load(query_file)
query_y, query_cam = f['label'], f['cam']
data = utils.extract_data_from_lst(f['lst'],input_shape=params1['crop_shape'])
query_f1 = fea_mode1.predict(data)
data = utils.extract_data_from_lst(f['lst'],input_shape=params2['crop_shape'])
query_f2 = fea_mode2.predict(data)

f = np.load(test_file)
test_y, test_cam = f['label'], f['cam']
data = utils.extract_data_from_lst(f['lst'],input_shape=params1['crop_shape'])
test_f1 = fea_mode1.predict(data)
data = utils.extract_data_from_lst(f['lst'],input_shape=params2['crop_shape'])
test_f2 = fea_mode2.predict(data)

cos_dist1 = cdist(query_f1, test_f1, metric='cosine')
cos_dist2 = cdist(query_f1, test_f1, metric='cosine')

cos_dist = cos_dist1 + cos_dist2

evaluation.evaluate(cos_dist,query_y,test_y,query_cam,test_cam)







