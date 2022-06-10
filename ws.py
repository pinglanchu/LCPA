import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from LCPA import method
from SPM import SPM
from train_test_split import DivideNet
from Sigma_c import cal_sigma_c
from random_segment_train_test import random_get_train_test
from scipy import optimize


def getMaxEG(n, m):
    if 2 * m >= n:
        max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
    else:
        max_EG = np.sqrt(2*m*n)
    return max_EG


def f1(x, a, b):
    return a * x + b


def cal_all_values(neighbors):
    LCPA_value = []
    sigma_c = []
    delta1 = []
    for p_link in np.linspace(0.1, 0.9, 9):
        ws = nx.watts_strogatz_graph(300, neighbors, p_link)
        adj_mat = nx.adjacency_matrix(ws).todense()
        n = adj_mat.shape[0]
        m = np.sum(np.triu(adj_mat))
        result = method(adj_mat, 0.3, 0.1)
        LCPA_value.append(result)

        train, test = DivideNet(adj_mat, 0.1)
        # train, test = random_get_train_test(adj_mat, 0.01)
        A_R, A_T = DivideNet(train, 0.1)
        sigma = SPM(train, test, A_R, A_T)
        # sigma = cal_sigma_c(train, test)
        sigma_c.append(sigma)

        eigvals = np.linalg.eigvals(adj_mat)
        energy = sum(abs(eigvals))

        delta1.append(getMaxEG(n, m) / energy)

    delta2 = [delta1[i] * sigma_c[i] for i in range(9)]
    return LCPA_value, sigma_c, delta1, delta2


def get_delta():
    LCPA_value_arr = np.zeros((10, 9))
    sigma_c_arr = np.zeros((10, 9))
    delta1_arr = np.zeros((10, 9))
    delta2_arr = np.zeros((10, 9))
    for repeat in range(10):
        LCPA_value, sigma_c, delta1, delta2 = cal_all_values(12)
        LCPA_value_arr[repeat, :] = LCPA_value
        sigma_c_arr[repeat, :] = sigma_c
        delta1_arr[repeat, :] = delta1
        delta2_arr[repeat, :] = delta2
    return delta1_arr, delta2_arr, LCPA_value_arr


def get_sigmac():
    LCPA_value_arr = np.zeros((10, 9))
    sigma_c_arr = np.zeros((10, 9))
    for repeat in range(10):
        print('repeat=', repeat)
        LCPA_value, sigma_c, delta1, delta2 = cal_all_values(12)
        LCPA_value_arr[repeat, :] = LCPA_value
        sigma_c_arr[repeat, :] = sigma_c
    return sigma_c_arr, LCPA_value_arr


# delta1_arr, delta2_arr, LCPA_value_arr = get_delta()
# print(np.mean(delta1_arr, axis=0))
# print(np.mean(delta2_arr, axis=0))
# # #
#
# markers = ['^', 'P', 'X', 'v', 'p', '+', 'o', 's', '*']
# p_lst = np.linspace(0.1, 1, 10)
#
# a, b = optimize.curve_fit(f1, np.mean(delta1_arr, axis=0), np.mean(LCPA_value_arr, axis=0))[0]
# print('a', a, 'b', b)
# for i in range(9):
#     plt.scatter(np.mean(delta1_arr[:, i]), np.mean(LCPA_value_arr[:, i]), marker=markers[i],
#                 label='p=%.2f' % p_lst[i], s=70)
# plt.plot(np.mean(delta1_arr, axis=0), a * np.mean(delta1_arr, axis=0) + b, c='black')
# plt.legend(frameon=False)
# plt.xlabel(r'$\delta_{se}$')
# plt.ylabel('predictability')
# plt.show()


# a, b = optimize.curve_fit(f1, np.mean(delta2_arr, axis=0), np.mean(LCPA_value_arr, axis=0))[0]
# print('a', a, 'b', b)
# for i in range(9):
#     plt.scatter(np.mean(delta2_arr[:, i]), np.mean(LCPA_value_arr[:, i]), marker=markers[i],
#                 label='p=%.2f' % p_lst[i], s=70)
# plt.plot(np.mean(delta2_arr, axis=0), a * np.mean(delta2_arr, axis=0) + b, c='black')
# plt.legend(frameon=False)
# plt.xlabel(r'$\delta_{sesc}$')
# plt.ylabel('predictability')
# plt.show()










