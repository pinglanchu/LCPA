import numpy as np
from train_test_split import DivideNet
from random_segment_train_test import random_get_train_test
import scipy.io
import os


def method(A, q, p):
    n = A.shape[0]

    train, test = DivideNet(A, q)  # 将网络划分为训练集和测试机 (划分后保持网络的连通性)
    # train, test = random_get_train_test(A, q)  # （网络随机划分训练集和测试集）

    L = int(np.sum(np.triu(test)))
    A_R, A_T = DivideNet(train, p)  # 在训练集中，将网络分为剩余网络和摄动网络
    # A_R, A_T = random_get_train_test(train, p)  # （网络随机划分训练集和测试集）

    AR_eigvals, AR_eigvecs = np.linalg.eig(A_R)

    A_R_zeroindex = np.argwhere(np.triu(A_R) + np.tril(np.ones((n, n))) == 0)
    A_R_zeroindex = [list(x) for x in A_R_zeroindex]

    A_T_oneindex = np.argwhere(np.triu(A_T) == 1)
    A_T_oneindex = [list(x) for x in A_T_oneindex]

    modify_index = [x for x in A_R_zeroindex if x not in A_T_oneindex]

    delta_eigvals = []
    appro_eigvecs = []

    for i in range(n):
        temp = AR_eigvecs[:, i].T * A_T * AR_eigvecs[:, i] / (AR_eigvecs[:, i].T * AR_eigvecs[:, i])
        delta_eigvals.append(temp)
        appro_eigvecs.append(np.real(AR_eigvals[i]) + np.real(temp[0, 0]))

    A_tilde = np.zeros((n, n))

    for i in range(n):
        A_tilde += (np.real(AR_eigvals[i]) + np.real(delta_eigvals[i][0, 0])) * (np.real(AR_eigvecs[:, i]) * np.real(AR_eigvecs[:, i].T))

    for index in modify_index:
        modify_value = A_tilde[index[0], index[1]] / (q + A_tilde[index[0], index[1]] * (1 - q))
        A_tilde[index[0], index[1]] = modify_value
        A_tilde[index[1], index[0]] = modify_value

    zero_index = np.argwhere(np.triu(train) + np.tril(np.ones((n, n))) == 0)
    zero_index = list(zero_index)

    zero_score = dict()

    for index in zero_index:
        zero_score[(index[0], index[1])] = A_tilde[index[0], index[1]]

    top_sorted_score = sorted(zero_score.items(), key=lambda x: x[1])[:-L-1:-1]

    num = 0

    for i in range(L):
        temp = top_sorted_score[i][0]
        if test[temp[0], temp[1]] == 1:
            num += 1
    return num / L


# 在最优参数下，10次独立执行的均值(q, p)
# parameters = [(0.2556, 0.1), (0.4889, 0.1), (0.1, 0.2556), (0.1, 0.1), (0.1, 0.1), (0.3333, 0.1), (0.2556, 0.1778),
#               (0.6444, 0.1), (0.1778, 0.1778), (0.4111, 0.1), (0.1778, 0.2556), (0.2556, 0.1778)]
#
# files = os.listdir('./Data')
# for m in range(12):
#     print(files[m])
#     precision_lst = []
#     for repeats in range(50):
#         q, p = parameters[m]
#         try:
#             data = scipy.io.loadmat('./Data/' + files[m])
#             A = data['A'].todense()
#             A = np.mat(A)
#             result = method(A, q, p)
#             precision_lst.append(result)
#             print('q:{}, p:{}, 预测精度{}'.format(q, p, result))
#             # precision_lst.append(result)
#         except IndexError as e:
#             pass
#         continue
#     print(np.mean(precision_lst))


# 计算随着q变化，链路预测精度值的变化
# with open('LCPA_pH_result.txt', 'w') as f:
#     for m in range(12):
#         print(files[m])
#         f.write(str(files[m]) + '\n')
#         p = parameters[m][1]
#         mean_precison_lst = []
#         std_lst = []
#         q_lst = []
#         for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
#             temp_q = 0
#             k = 0
#             temp_precision = []
#             for repeats in range(10):
#                 try:
#                     data = scipy.io.loadmat('./Data/' + files[m])
#                     A = data['A'].todense()
#                     A = np.mat(A)
#                     result = method(A, q, p)
#                     print('预测精度', result)
#                     temp_precision.append(result)
#                     temp_q += q
#                     k += 1
#                 except IndexError as e:
#                     pass
#                 continue
#             if k:
#                 q_lst.append(temp_q / k)
#                 mean_precison_lst.append(np.mean(temp_precision))
#                 std_lst.append(np.std(temp_precision))
#
#         f.write('mean' + str(mean_precison_lst) + '\n')
#         f.write('std' + str(std_lst) + '\n')
#         f.write('q' + str(q_lst) + '\n')
