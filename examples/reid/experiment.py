import numpy as np
import cotrain
import numpy as np
import utils
from scipy.spatial.distance import cdist
import evaluation
from keras.models import Model
import model



name_model1 = 'xception'
name_model2 = 'resnet50'

params1 = {
    'input_shape': (336,336,3),
    'crop_shape': (299,299,3),
    'epochs': 70,
    'steps_per_epoch': 250,
    'batch_size': 16,
}

params2 = {
    'input_shape': (256,256,3),
    'crop_shape': (224,224,3),
    'epochs': 70,
    'steps_per_epoch': 250,
    'batch_size': 16,
}

train_lst = '../data/train_lst.npz'
untrain_lst = '../data/untrain_lst.npz'
#model1,model2 = cotrain.cotrain(train_lst,untrain_lst,name_model1,name_model2,params1,params2)
#model1.save_weights('../models/cotrain_xception.h5')
#model2.save_weights('../models/cotrain_resnet.h5')

model1 = model.get_model(model_name=name_model1)
model2 = model.get_model(model_name=name_model2)
model1.load_weights('../models/cotrain_xception.h5')
model2.load_weights('../models/cotrain_resnet.h5')
fea_mod1 = Model(model1.input,model1.layers[-1].input)
fea_mod2 = Model(model2.input,model2.layers[-1].input)

query_file = '../data/query.lst.npz'
test_file  = '../data/test.lst.npz'
f = np.load(query_file)
query_y, query_cam = f['label'], f['cam']
query_f1 = utils.extract_feature(fea_mod1,f['lst'])
query_f2 = utils.extract_feature(fea_mod2,f['lst'])

f = np.load(test_file)
test_y, test_cam = f['label'], f['cam']
test_f1 = utils.extract_feature(fea_mod1,f['lst'])
test_f2 = utils.extract_feature(fea_mod2,f['lst'])

cos_dist1 = cdist(query_f1, test_f1, metric='cosine')
cos_dist2 = cdist(query_f2, test_f2, metric='cosine')

cos_dist = cos_dist1 + cos_dist2

evaluation.evaluate(cos_dist,query_y,test_y,query_cam,test_cam)







