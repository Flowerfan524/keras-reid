import numpy as np

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
    np.savez('../data/untrain)lst', lst=lst_untrain, label=y_untrain)

train_lst = '../data/train.lst.npz'
split_data(train_lst)
