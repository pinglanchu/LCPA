import numpy as np
from train_test_split import DivideNet, spones


def CN(A, q):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    sim = np.dot(train, train)

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def RA(A, q):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    add_row_matrix = np.tile(train.sum(axis=1).reshape(n, 1), (1, train.shape[0]))
    sim = np.true_divide(train, add_row_matrix)
    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0
    sim = train * sim

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def AA(A, q):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    add_row_matrix = np.tile(np.log(train.sum(axis=1).reshape(n, 1)), (1, train.shape[0]))
    sim = np.true_divide(train, add_row_matrix)

    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0

    sim = train * sim

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def Salton(A, q):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    tempdeg = np.tile(np.sqrt(train.sum(axis=1).reshape(n, 1)), (1, train.shape[0]))
    tempdeg = tempdeg * tempdeg.T

    sim = np.dot(train, train)
    sim = np.true_divide(sim, tempdeg)
    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def Jaccard(A, q):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    sim = np.dot(train, train)
    deg_row = np.tile(train.sum(axis=0), (train.shape[0], 1))
    deg_row = deg_row * sim  # spones(sim)
    deg_row = np.triu(deg_row) + np.triu(deg_row.T)
    sim = np.true_divide(sim, deg_row * sim - sim)

    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def PA(A, q):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    deg_row = np.sum(train, axis=1).reshape(n, 1)
    sim = np.dot(deg_row, deg_row.T)

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def HPI(A, q):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    sim = np.dot(train, train)
    deg_row = np.tile(train.sum(axis=0), (train.shape[0], 1))
    deg_row = deg_row * spones(sim)
    deg_row = np.minimum(deg_row, deg_row.T)
    sim = np.true_divide(sim, deg_row)

    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def HDI(A, q):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    sim = np.dot(train, train)
    deg_row = np.tile(train.sum(axis=0), (train.shape[0], 1))
    deg_row = deg_row * spones(sim)
    deg_row = np.maximum(deg_row, deg_row.T)
    sim = np.true_divide(sim, deg_row)

    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def LHN_I(A, q):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    sim = np.dot(train, train)
    deg = np.sum(train, axis=1).reshape(n, 1)
    deg = np.dot(deg, deg.T)
    sim = np.true_divide(sim, deg)

    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def Katz(A, q, lam):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    sim = np.linalg.inv(np.eye(train.shape[0]) - lam * train)

    sim = sim - np.eye(train.shape[0])

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def RWR(A, q, lam):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    deg = np.tile(np.sum(train, axis=1).reshape(n, 1), (1, train.shape[1]))
    train = np.true_divide(train, deg)
    I = np.eye(train.shape[0])
    sim = (1-lam) * np.dot(np.linalg.inv(I - lam * train.T), I)
    sim = sim + sim.T

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def LRW(A, q, steps):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    deg = np.tile(np.sum(train, axis=1).reshape(n, 1), [1, train.shape[1]])
    M = np.sum(train)
    train = np.true_divide(train, deg)
    I = np.eye(train.shape[0])
    sim = I
    stepi = 0
    while stepi < steps:
        sim = np.dot(train.T, sim)
        stepi += 1
    sim = sim.T * deg / M
    sim = sim + sim.T

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


def SRW(A, q, steps):
    n = A.shape[0]
    m = np.sum(np.triu(A))
    L = int(np.floor(m * q))
    train, test = DivideNet(A, q)
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))))
    zero_index = list(zero_index)

    deg = np.tile(np.sum(train, axis=1).reshape(n, 1), [1, train.shape[1]])
    M = np.sum(train)
    train = np.true_divide(train, deg)
    I = np.eye(train.shape[0])
    tempsim = I
    stepi = 0
    sim = np.zeros((n, n))
    while stepi < steps:
        tempsim = np.dot(train.T, tempsim)
        stepi += 1
        sim += tempsim

    sim = sim.T * deg / M
    sim = sim + sim.T

    zero_score = dict()
    for index in zero_index:
        zero_score[(index[0], index[1])] = sim[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0
    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L

