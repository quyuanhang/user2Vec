import sys 
from time import time

import math
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
from torch import nn, LongTensor, FloatTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

TRAIN_FILE_PATH = 'data/interview.train'
TEST_FILE_PATH = 'data/interview.test'
USE_GPU = torch.cuda.is_available()
USE_GPU = False
BATCH_SIZE = 256
N_EPOCH = 50
LERANING_RATE = 0.001
NEG_SAMPLE = 4
TOP_K = 10

def load_train(filename):
    '''
    Read .rating file and Return dok matrix.
    The first line of .rating file is: n_job\t n_geek
    '''
    raw_frame = pd.read_table(filename, header=None, sep='\t')
    # raw_frame = raw_frame.iloc[:10000]
    n_job = raw_frame[0].max()
    n_geek = raw_frame[1].max()
    # Construct matrix
    mat = sp.dok_matrix((n_job+1, n_geek+1), dtype=np.float32)
    for i, row in raw_frame.iterrows():
        user, item, rating = row
        mat[user, item] = rating
    return mat

def get_train_instances(train, num_negatives):
    n_job, n_geek = train.shape
    data = []
    # num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        data.append([(u, i), 1])
        # negative instances
        for _ in range(num_negatives):
            j = np.random.randint(n_geek)
            while bool(train[u, j]):
                j = np.random.randint(n_geek)
            data.append([(u, j), 0])
    data = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return data

def load_test(filename):
    raw_frame = pd.read_table(filename)
    return raw_frame.values

class MLP(nn.Module):
    def __init__(self, n_job, n_geek, layers_dim):
        super(MLP, self).__init__()
        # self.layers = []
        self.job_emb = nn.Embedding(n_job, int(layers_dim[0] / 2))
        self.geek_emb = nn.Embedding(n_geek, int(layers_dim[0] / 2))
        # for i in range(len(layers_dim) - 1):
        #     layer = nn.Sequential(
        #         nn.Linear(layers_dim[i], layers_dim[i+1]),
        #         nn.ReLU()
        #     )
        #     self.layers.append(layer))
        # self.out = nn.Sigmoid()

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, job, geek):
        job = self.job_emb(job)
        geek = self.geek_emb(geek)
        x = torch.cat((job, geek), 1)
        # for layer in self.layers:
        #     x = layer(x)
        x = self.out(x)
        return x

def get_model(shape, layers_dim = [64, 32, 16, 8, 1]):
    n_job, n_geek = shape    
    model = MLP(n_job, n_geek, layers_dim)
    if USE_GPU:
        model.cuda()
    return model

def train_model(model, n_epoch, data, learning_rate):
    # 定义loss和optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(n_epoch):
        print('epoch {} \n'.format(epoch + 1) + '*' * 10)
        running_loss = 0.0
        for i, sample in enumerate(data, 1):
            feature, label = sample
            job, geek = feature
            job = Variable(LongTensor(job))
            geek = Variable(LongTensor(geek))
            label = label.view(label.shape[0], 1)
            label = label.float()
            if USE_GPU:
                job = job.cuda()
                geek = geek.cuda()
                label = label.cuda()
            # 前向传播计算损失
            out = model(job, geek)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            # 后向传播计算梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 监控指标
            if i % 300 == 0:
                print('[{}/{}] Loss: {:.6f}'.format(
                    epoch + 1, n_epoch, running_loss / (BATCH_SIZE * i)))

def get_prediction(model, test_data):
    predictions = []
    with torch.no_grad():
        for sample in test_data:
            job = sample[0]
            pos_geek = sample[1]
            geeks = sample[1:]
            job_tensor = Variable(LongTensor([job] * len(geeks)))
            geeks_tensor = Variable(LongTensor(geeks))
            if USE_GPU:
                job_tensor = job_tensor.cuda()
                geeks_tensor = geeks_tensor.cuda()
            scores = model.forward(job_tensor, geeks_tensor)
            scores = scores.cpu()
            scores = scores.numpy()
            predictions.append(scores)
    return predictions

def evaluate(predictions, top_K):
    hits, ndcgs, aucs = [], [], []
    for scores in predictions:
        hr, ndcg, auc = eval_one_rating(scores, top_K)
        hits.append(hr)
        ndcgs.append(ndcg)  
        aucs.append(auc)
    return np.array(hits).mean(), np.array(ndcgs).mean(), np.array(aucs).mean()

def eval_one_rating(scores, top_K):
    pos_score = scores[0]
    false_posi = scores[scores >= pos_score]
    pos_rank = len(false_posi)
    hr = get_hit_ratio(pos_rank, top_K)
    ndcg = get_NDCG(pos_rank)
    auc = get_AUC(pos_rank, len(scores))
    return hr, ndcg, auc

def get_hit_ratio(pos_rank, top_K):
    return int(pos_rank < top_K)

def get_NDCG(pos_rank):
    return math.log(2) / math.log(pos_rank + 2)

def get_AUC(pos_rank, n):
    auc = 1 - pos_rank / n
    return auc


if __name__ == '__main__':
    USE_GPU = eval(sys.argv[1])
    N_EPOCH = eval(sys.argv[2])
    if USE_GPU:
        print('using GPU')
    train_data = load_train(TRAIN_FILE_PATH)
    train_instance = get_train_instances(train_data, NEG_SAMPLE)
    model = get_model(train_data.shape)
    train_model(model, N_EPOCH, train_instance, LERANING_RATE)

    test_data = load_test(TEST_FILE_PATH)
    predictions = get_prediction(model, test_data)
    hr, ndcg, auc = evaluate(predictions, TOP_K)
    print('hit ratio: {:.6f} NDCG: {:.6f}, AUC: {:.6f}'.format(hr, ndcg, auc))

