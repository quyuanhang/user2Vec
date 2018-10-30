import sys
import argparse
from tqdm import tqdm
from collections import Iterable
import pandas as pd
import numpy as np


def count_degree(frame, col):
    user = frame.columns[col]
    user_degree_series = frame.iloc[:, col]
    user_degree_frame = pd.DataFrame(user_degree_series.value_counts())
    user_degree_frame.columns = ['degree']
    user_degree_frame = pd.merge(frame, user_degree_frame,
                                 left_on=user, right_index=True)
    return user_degree_frame


def filter_old(frame, N=0):
    frame = count_degree(frame, 0)
    frame = count_degree(frame, 1)
    old_frame = frame[(frame['degree_x'] >= N) & (frame['degree_y'] >= N)]
    # print('rest users', len(set(old_frame.iloc[:, 0])))
    # print('rest items', len(set(old_frame.iloc[:, 1])))
    # print('rest matches', len(old_frame))
    return old_frame.iloc[:, :2]


def iter_filter_old(frame, N=5, M=5, step=100):
    for i in range(step):
        frame = filter_old(frame.iloc[:, :2], N)
        frame = count_degree(frame, 0)
        frame = count_degree(frame, 1)
        # print(frame.head())
        if frame['degree_x'].min() >= N and frame['degree_y'].min() >= M:
            # print(frame.describe())
            break
    return frame.iloc[:, :2]


def train_test_split(train_frame, test_frame):

    print('train test \n', len(train_frame), len(test_frame))
    n_geek = train_frame['geek'].max()
    print('geek num \n', n_geek)

    neg_frame = pd.DataFrame()
    train_frame.index = train_frame['job']
    test_frame = test_frame.drop_duplicates(subset='job')
    test_frame.index = test_frame['job']

    for job in tqdm(set(test_frame['job'])):
        job_action_train = train_frame.loc[job, ['job', 'geek']]
        train = job_action_train['geek']
        train = set(train) if isinstance(train, Iterable) else set([train])
        job_action_test = test_frame.loc[job, ['job', 'geek']]
        test = job_action_test['geek']
        test = set(train) if isinstance(test, Iterable) else set([test])
        positive = train | test
        negatives = list()
        while len(negatives) < 100:
            idxs = np.random.randint(0, n_geek, 100)
            geeks = [geek for geek in idxs if geek not in positive][:(100-len(negatives))]
            negatives.extend(geeks)
        negative_sample = pd.Series(negatives)
        negative_sample.index = range(100)
        negative_sample.name = job
        neg_frame = neg_frame.append(negative_sample)
    neg_frame = neg_frame.astype(int)
    return train_frame, test_frame, neg_frame


def convert_data(*args):
    train_frame, test_frame = args
    job_dict = {k:v for v, k in enumerate(train_frame['job'].drop_duplicates())}
    geek_dict = {k:v for v, k in enumerate(train_frame['geek'].drop_duplicates())}
    print('n_job n_geek')
    print(len(job_dict), len(geek_dict))

    train_frame['job'] = train_frame['job'].map(lambda x: job_dict[x])
    train_frame['geek'] = train_frame['geek'].map(lambda x: geek_dict[x])
    train_frame['rating'] = 3

    test_frame = pd.DataFrame([
        [job_dict[job], geek_dict[geek]]
        for job, geek in test_frame.values
        if job in job_dict and geek in geek_dict],
        columns=['job', 'geek'])
    test_frame.index = test_frame['job']
    print(len(train_frame), len(test_frame))

    return train_frame, test_frame, job_dict, geek_dict

def train_frame_plus(train_frame, job_add, geek_add, job_dict, geek_dict):
    print('interview num', len(train_frame))
    _job_add = pd.DataFrame([
        [job_dict[job], geek_dict[geek], 1]
        for job, geek in job_add.values
        if job in job_dict and geek in geek_dict],
        columns=['job', 'geek', 'rating'])
    print('job posi num', len(_job_add))
    _job_add = _job_add.sample(frac=(len(train_frame)/len(_job_add)))
    print('job posi num', len(_job_add))
    train_frame_p = pd.concat((train_frame, _job_add), sort=False)
    _geek_add = pd.DataFrame([
        [job_dict[job], geek_dict[geek], 2]
        for job, geek in geek_add.values
        if job in job_dict and geek in geek_dict],
        columns=['job', 'geek', 'rating'])
    print('geek posi num', len(_geek_add))
    _geek_add = _job_add.sample(frac=(len(train_frame)/len(_geek_add)))
    print('geek posi num', len(_geek_add))
    train_frame_p = pd.concat((train_frame_p, _geek_add), sort=False)
    return train_frame_p


def dataFormat(d):
    if type(d) == float:
        return str(int(d))
    else:
        return str(d)

def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--datain', nargs='?', default='ml-1m')
    parser.add_argument('--dataout', default='interview')
    parser.add_argument('--t', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    T = args.t
    DATA_IN = args.datain
    DATA_OUT = args.dataout

    # read data
    train_frame = (pd.read_table('{}.train'.format(DATA_IN), header=None)
        .iloc[:, :2].dropna().applymap(dataFormat))
    train_frame.columns = ['job', 'geek']
    test_frame = (pd.read_table('{}.test'.format(DATA_IN), header=None)
                  .iloc[:, :2].dropna().applymap(dataFormat))
    test_frame.columns = ['job', 'geek']

    train_frame = iter_filter_old(train_frame, T, T)
    train_frame, test_frame, job_dict, geek_dict = convert_data(train_frame, test_frame)
    train_frame, test_frame, neg_frame = train_test_split(train_frame, test_frame)
    train_frame.to_csv('data/{}.train'.format(DATA_OUT), index=False, header=False, sep='\t')
    neg_frame.to_csv('data/{}.test'.format(DATA_OUT), index=False, header=False, sep='\t')

    job_add = (pd.read_table('{}_job.txt'.format(DATA_IN), header=None)
        .dropna().applymap(dataFormat))
    geek_add = (pd.read_table('{}_geek.txt'.format(DATA_IN), header=None)
        .dropna().applymap(dataFormat))
    train_frame = train_frame_plus(train_frame, job_add, geek_add, job_dict, geek_dict)

    train_frame.to_csv('data/{}.multi.train'.format(DATA_OUT), index=False, header=False, sep='\t')
