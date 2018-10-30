import numpy as np
import pandas as pd
from scipy import sparse
from torch.utils.data import DataLoader

def load_train(filename):
    '''
    Read .rating file and Return dok matrix.
    The first line of .rating file is: n_job\t n_geek
    '''
    raw_frame = pd.read_table(filename, header=None, sep='\t')
    print(raw_frame.head())
    # raw_frame = raw_frame.iloc[:10000]
    n_job = raw_frame[0].max()
    n_geek = raw_frame[1].max()
    # Construct matrix
    mat = sparse.dok_matrix((n_job+1, n_geek+1), dtype=np.float32)
    for i, row in raw_frame.iterrows():
        user, item, rating = row
        mat[user, item] = max(rating, mat.get((user, item), 0))
    print(mat.shape)
    return mat

def get_train_instances(train, num_negatives, batch_size):
    n_job, n_geek = train.shape
    data = []
    # num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        data.append([u, i, 1])
        # negative instances
        for _ in range(num_negatives):
            j = np.random.randint(n_geek)
            while bool(train[u, j]):
                j = np.random.randint(n_geek)
            data.append([u, j, 0])
    # import pdb
    # pdb.set_trace()
    data = np.random.permutation(data)
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data

def get_train_instances_multi(train, num_negatives, batch_size):
    label_dic = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [0, 0, 1],
        3: [1, 1, 1]
    }
    n_job, n_geek = train.shape
    data = []
    for (u, i), r in train.items():
        # positive instance
        label = label_dic[r]
        data.append([u, i, *label])
        # negative instances
        for _ in range(num_negatives):
            j = np.random.randint(n_geek)
            while bool(train[u, j]):
                j = np.random.randint(n_geek)
            data.append([u, j, 0, 0, 0])
    data = np.random.permutation(data)
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data

def load_test(filename):
    raw_frame = pd.read_table(filename, header=None)
    print(raw_frame.head())
    return raw_frame.values
