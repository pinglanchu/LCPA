import numpy as np


def SPM(train, test, A_R, A_T):
    n = train.shape[0]

    L = int(np.sum(np.triu(test)))

    AR_eigvals, AR_eigvecs = np.linalg.eig(A_R)

    delta_eigvals = []
    appro_eigvecs = []

    # print('first loop')
    for i in range(n):
        temp = AR_eigvecs[:, i].T * A_T * AR_eigvecs[:, i] / (AR_eigvecs[:, i].T * AR_eigvecs[:, i])
        delta_eigvals.append(temp)
        appro_eigvecs.append(np.real(AR_eigvals[i]) + np.real(temp[0, 0]))

    # print('second loop')
    appro_vals = [np.real(AR_eigvals[i]) + np.real(delta_eigvals[i][0, 0]) for i in range(n)]
    A_tilde = np.dot(AR_eigvecs, np.diag(appro_vals)).dot(AR_eigvecs.T)

    # print('third loop')
    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))) == 0)

    zero_score = dict()

    # print('fourth loop')
    for index in zero_index:
        zero_score[(index[0], index[1])] = A_tilde[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    # print('fifth loop')
    num = 0
    for i in range(L):
        temp_index = top_sorted_score[i][0]
        if test[temp_index[0], temp_index[1]] == 1:
            num += 1
    # print('=' * 20)
    return num / L
    # np.linalg.eigvals(train+test), np.array(appro_eigvecs)
