#coding=utf-8
import sys
import argparse
import torch
# from models import MF, GMF, MLP
from MLP import MLP
from GMF import GMF
from MTL import MTL
from MF import MF
from evaluate import evaluate
from dataSet import load_train, load_test, get_train_instances, get_train_instances_multi


def parse_args():
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--reg', type=float, default=0,
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--model', default='MF')
    return parser.parse_args()



def train_model(model, n_epoch, train_instance, train_data, test_data, learning_rate):
    # 定义loss和optimizer
    predictions = model.predict(test_data)
    hr, ndcg, auc = evaluate(predictions, TOP_K)
    print('init')
    print('hit ratio: {:.6f} NDCG: {:.6f}, AUC: {:.6f}'.format(hr, ndcg, auc))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * 10, weight_decay=REG)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=REG)
    for epoch in range(n_epoch):
        print('epoch {} \n'.format(epoch + 1) + '*' * 10)
        running_loss = 0.0
        if epoch > 10:
            optimizer = optimizer2
        for i, sample in enumerate(train_instance, 1):
            batch_loss = model.batch_fit(model, optimizer, sample)
            running_loss += batch_loss
            if i % (len(train_instance) // 2) == 0:
                print('[{}/{}] Loss: {:.6f}'.format(
                    epoch + 1, n_epoch, running_loss / (BATCH_SIZE * i)))
        predictions = model.predict(test_data)
        hr, ndcg, auc = evaluate(predictions, TOP_K)
        print('hit ratio: {:.6f} NDCG: {:.6f}, AUC: {:.6f}'.format(hr, ndcg, auc))


if __name__ == '__main__':

    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        print('using GPU')

    args = parse_args()
    TRAIN_FILE_PATH = 'data/{}.train'.format(args.dataset)
    MULTI_TRAIN_FILE_PATH = 'data/{}.multi.train'.format(args.dataset)
    TEST_FILE_PATH = 'data/{}.test'.format(args.dataset)
    BATCH_SIZE = args.batch_size
    N_EPOCH = args.epochs
    LERANING_RATE = args.lr
    REG = args.reg
    NEG_SAMPLE = args.num_neg
    TOP_K = args.topk
    MODEL = args.model

    test_data = load_test(TEST_FILE_PATH)

    if MODEL == 'MTL':
        train_data = load_train(MULTI_TRAIN_FILE_PATH)
        train_instance = get_train_instances_multi(train_data, NEG_SAMPLE, BATCH_SIZE)
        model = MTL(train_data.shape[0], train_data.shape[1], 36)
    elif MODEL == 'GMF':
        train_data = load_train(TRAIN_FILE_PATH)
        train_instance = get_train_instances(train_data, NEG_SAMPLE, BATCH_SIZE)
        model = GMF(train_data.shape[0], train_data.shape[1], 32)
    elif MODEL == 'MLP':
        train_data = load_train(TRAIN_FILE_PATH)
        train_instance = get_train_instances(train_data, NEG_SAMPLE, BATCH_SIZE)
        model = MLP(train_data.shape[0], train_data.shape[1], 32)
    else:
        print('no model selected')
        sys.exit(0)

    if USE_GPU:
        model.cuda()

    # from visualize import make_dot
    # g = make_dot(model(LongTensor([0]).cuda(), LongTensor([0]).cuda()))
    # g.render('model.pdf', view=False)

    train_model(model, N_EPOCH, train_instance, train_data, test_data, LERANING_RATE)
    predictions = model.predict(test_data)
    hr, ndcg, auc = evaluate(predictions, TOP_K)
    print('hit ratio: {:.6f} NDCG: {:.6f}, AUC: {:.6f}'.format(hr, ndcg, auc))

