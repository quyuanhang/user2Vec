import sys
import argparse
from tqdm import tqdm
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


def filter_old(frame, N=0, M=100000):
    frame = count_degree(frame, 0)
    frame = count_degree(frame, 1)
    old_frame = frame[(frame['degree_x'] >= N) & (frame['degree_y'] >= N)]
    # print('rest users', len(set(old_frame.iloc[:, 0])))
    # print('rest items', len(set(old_frame.iloc[:, 1])))
    # print('rest matches', len(old_frame))
    return old_frame.iloc[:, :2]


def iter_filter_old(frame, N=10, M=5, step=100):
    for i in range(step):
        frame = filter_old(frame.iloc[:, :2], N, M)
        frame = count_degree(frame, 0)
        frame = count_degree(frame, 1)
        # print(frame.head())
        if frame['degree_x'].min() >= N and frame['degree_y'].min() >= M:
            print(frame.describe())
            break
    return frame.iloc[:, :2]


def train_test_split(df):
    train_frame = pd.DataFrame()
    test_frame = pd.DataFrame()
    neg_frame = pd.DataFrame()
    test_set = set()

    print('all data')
    print(len(df))
    df.index = df['job']

    for job in tqdm(set(df['job'])):
        job_action = df.loc[job]
        train_frame = train_frame.append(job_action.iloc[:-1])
        test_frame = test_frame.append(job_action.iloc[-1])
        job, geek = job_action.iloc[-1]
        test_set.add((job, geek))
    train_frame = train_frame.reindex(columns=['job', 'geek', 'rating'])
    test_frame = test_frame.reindex(columns=['job', 'geek', 'rating'])

    print('train test')
    print(len(train_frame), len(test_frame))
    all_geek = np.array(list(set(train_frame['geek'])))
    print(len(all_geek))

    for job in tqdm(set(df['job'])):
        job_action = df.loc[job, ['job', 'geek']]
        # negative = list(all_geek - set(job_action['geek'].values))
        negatives = list()
        positive = set(job_action['geek'].values)
        while len(negatives) < 100:
            idxs = np.random.randint(0, len(all_geek), 100 - len(negatives))
            geeks = [geek for geek in all_geek[idxs] if geek not in positive]
            negatives.extend(geeks)
        negative_sample = pd.Series(negatives)
        negative_sample.index = range(100)
        negative_sample.name = job
        neg_frame = neg_frame.append(negative_sample)

    return train_frame, test_frame, neg_frame, test_set


def convert_data(*args):
    train_frame, test_frame, neg_frame = args
    job_dict = {k:v for v, k in enumerate(train_frame['job'].drop_duplicates())}
    geek_dict = {k:v for v, k in enumerate(train_frame['geek'].drop_duplicates())}
    print('n_job n_geek')
    print(len(job_dict), len(geek_dict))

    train_frame['job'] = train_frame['job'].map(lambda x: job_dict[x])
    train_frame['geek'] = train_frame['geek'].map(lambda x: geek_dict[x])
    train_frame['rating'] = 3

    test_frame = pd.DataFrame([
        [job_dict[job], geek_dict[geek]]
        for job, geek, rating in test_frame.values
        if job in job_dict and geek in geek_dict],
        columns=['job', 'geek'])
    test_frame.index = test_frame['job']
    test_frame = test_frame.drop_duplicates(subset='job')
    neg_frame.index = [job_dict[x] for x in neg_frame.index]
    # neg_frame = neg_frame.applymap(lambda x: x+1)
    neg_frame = neg_frame.applymap(lambda x: geek_dict.get(x, None))
    neg_frame = pd.concat([test_frame, neg_frame], axis=1, join='inner')
    print(neg_frame.head())
    print(len(train_frame), len(test_frame), len(neg_frame))

    return train_frame, test_frame, neg_frame, job_dict, geek_dict

def train_frame_plus(train_frame, job_add, geek_add, job_dict, geek_dict, test_set):
    print('interview num', len(train_frame))
    job_add = pd.DataFrame([
        [job_dict[job], geek_dict[geek], 1]
        for job, geek in job_add.values
        if job in job_dict and geek in geek_dict and (job, geek) not in test_set],
        columns=['job', 'geek', 'rating'])
    print('job posi num', len(job_add))
    train_frame = pd.concat((train_frame, job_add), sort=False)
    geek_add = pd.DataFrame([
        [job_dict[job], geek_dict[geek], 2]
        for job, geek in geek_add.values
        if job in job_dict and geek in geek_dict and (job, geek) not in test_set],
        columns=['job', 'geek', 'rating'])
    print('geek posi num', len(geek_add))
    train_frame = pd.concat((train_frame, geek_add), sort=False)
    return train_frame


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
    df = pd.read_table('{}.txt'.format(DATA_IN), header=None).dropna().applymap(dataFormat)
    df.columns = ['job', 'geek']
    df['rating'] = 2

    df = iter_filter_old(df, T, T)
    train_frame, test_frame, neg_frame, test_set = train_test_split(df)
    train_frame, test_frame, neg_frame, job_dict, geek_dict = convert_data(train_frame, test_frame, neg_frame)

    train_frame.to_csv('data/{}.train'.format(DATA_OUT), index=False, header=False, sep='\t')
    neg_frame.to_csv('data/{}.test'.format(DATA_OUT), index=False, header=False, sep='\t')

    job_add = pd.read_table('{}_job.txt'.format(DATA_IN), header=None).dropna().applymap(dataFormat)
    geek_add = pd.read_table('{}_geek.txt'.format(DATA_IN), header=None).dropna().applymap(dataFormat)
    train_frame = train_frame_plus(train_frame, job_add, geek_add, job_dict, geek_dict, test_set)

    train_frame.to_csv('data/{}.multi.train'.format(DATA_OUT), index=False, header=False, sep='\t')
