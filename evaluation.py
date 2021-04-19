"""
For SPMF, STNE, SMF and SIGNet
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy.io import loadmat
import networkx as nx
from sklearn.preprocessing import StandardScaler
import random as rd
from sklearn.exceptions import ConvergenceWarning
import sys
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)


DATA_DIR = 'data/'
EMB_DIR = 'output/'
DATASET_NAME = sys.argv[1]
REMAIN = 80
test_size = float(sys.argv[2])
DIM = 128  # Dimension of emb
LINK_PREDICTION = True

outEmb = np.load(EMB_DIR + '/%s-%g-out.npy' % (DATASET_NAME, test_size))
inEmb = np.load(EMB_DIR + '/%s-%g-in.npy' % (DATASET_NAME, test_size))

numNodes = outEmb.shape[0]
print(numNodes)

trainSet = np.loadtxt(DATA_DIR + DATASET_NAME + '/%s-%g-train.txt' % (DATASET_NAME, test_size), dtype=np.int)
lenTrainSet = trainSet.shape[0]
trainX = np.zeros((lenTrainSet, DIM * 4))
trainY = np.zeros((lenTrainSet, 1))
pos = 0
neg = 0
for i in range(lenTrainSet):
    edge = trainSet[i]
    u = edge[0]
    v = edge[1]
    if edge[2] > 0:
        trainY[i] = 1
        pos += 1
    else:
        trainY[i] = -1
        neg += 1
    trainX[i, : DIM] = outEmb[u]
    trainX[i, DIM: DIM * 2] = inEmb[u]
    trainX[i, DIM * 2: DIM * 3] = outEmb[v]
    trainX[i, DIM * 3: DIM * 4] = inEmb[v]
print('Pos-neg ratio of train set: ' + str(pos / neg))

testSet = np.loadtxt(DATA_DIR + DATASET_NAME + '/%s-%g-test.txt' % (DATASET_NAME, test_size), dtype=np.int)
# testSet = np.loadtxt('../testSet.txt', dtype=np.int)
lenTestSet = testSet.shape[0]
testX = np.zeros((lenTestSet, DIM * 4))
testY = np.zeros((lenTestSet, 1))
pos = 0
neg = 0
for i in range(lenTestSet):
    edge = testSet[i]
    u = edge[0]
    v = edge[1]
    if edge[2] > 0:
        testY[i] = 1
        pos += 1
    else:
        testY[i] = -1
        neg += 1
    testX[i, : DIM] = outEmb[u]
    testX[i, DIM: DIM * 2] = inEmb[u]
    testX[i, DIM * 2: DIM * 3] = outEmb[v]
    testX[i, DIM * 3: DIM * 4] = inEmb[v]
print('Pos-neg ratio of test set: ' + str(pos / neg))

lr = LogisticRegression()
lr.fit(trainX, trainY)
testYScore = lr.predict_proba(testX)[:, 1]
testYPred = lr.predict(testX)
AucScore = roc_auc_score(testY, testYScore, average='macro')
MacroF1Score = f1_score(testY, testYPred, average='macro')
Acc = accuracy_score(testY, testYPred)

print('---SIGN PREDICTION')
print('AUC: ' + str(AucScore))
print('Macro-F1: ' + str(MacroF1Score))


if LINK_PREDICTION:
    trainSet = np.loadtxt(DATA_DIR + DATASET_NAME + '/null-%s-%g-train.txt' % (DATASET_NAME, test_size), dtype=np.int)
    testSet = np.loadtxt(DATA_DIR + DATASET_NAME + '/null-%s-%g-test.txt' % (DATASET_NAME, test_size), dtype=np.int)

    train_x = np.zeros((trainSet.shape[0], DIM * 4))
    train_y = np.zeros((trainSet.shape[0], 1))

    train_x = np.zeros((trainSet.shape[0], DIM * 4))
    train_y = np.zeros((trainSet.shape[0], 1))

    for i in range(trainSet.shape[0]):
        edge = trainSet[i]
        u = edge[0]
        v = edge[1]
        train_x[i, : DIM] = outEmb[u]
        train_x[i, DIM: DIM * 2] = inEmb[u]
        train_x[i, DIM * 2: DIM * 3] = outEmb[v]
        train_x[i, DIM * 3:] = inEmb[v]
        train_y[i] = 0
    train_x = np.vstack((trainX, train_x))
    train_y = np.vstack((trainY, train_y))

    test_x = np.zeros((testSet.shape[0], DIM * 4))
    test_y = np.zeros((testSet.shape[0], 1))

    for i in range(testSet.shape[0]):
        edge = testSet[i]
        u = edge[0]
        v = edge[1]
        test_x[i, : DIM] = outEmb[u]
        test_x[i, DIM: DIM * 2] = inEmb[u]
        test_x[i, DIM * 2: DIM * 3] = outEmb[v]
        test_x[i, DIM * 3:] = inEmb[v]
        test_y[i] = 0
    test_x = np.vstack((testX, test_x))
    test_y = np.vstack((testY, test_y))

    # ss = StandardScaler()
    # train_x = ss.fit_transform(train_x)
    # test_x = ss.fit_transform(test_x)
    # lr = LogisticRegressionCV(fit_intercept=True, max_iter=100, multi_class='multinomial', Cs=np.logspace(-2, 2, 20),
    #                           cv=2, penalty="l2", solver="lbfgs", tol=0.01)
    lr = LogisticRegression()

    def afunc(lr, train_x, train_y):
        lr.fit(train_x, train_y.ravel())
        return lr


    lr = afunc(lr, train_x, train_y)
    pred_prob = lr.predict_proba(test_x)
    pred_label = lr.predict(test_x)
    test_y = test_y[:, 0]
    AucScore = roc_auc_score(test_y, pred_prob, average='macro', multi_class='ovo')
    MacroF1Score = f1_score(test_y, pred_label, average='macro')

    print('---LINK PREDICTION')
    print('AUC: ' + str(AucScore))
    print('Macro-F1: ' + str(MacroF1Score))
