import networkx as nx
import numpy as np
from LCPA import method
from SPM import SPM
from train_test_split import DivideNet


def getMaxEG(n, m):
    if 2 * m >= n:
        max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
    else:
        max_EG = np.sqrt(2*m*n)
    return max_EG


def f1(x, a, b):
    return a * x + b


def cal_all_values():
    sigma_c = []
    delta1 = []
    for p_link in [0.05, 0.15, 0.25, 0.45]:
        er = nx.erdos_renyi_graph(300, p_link)
        adj_mat1 = nx.adjacency_matrix(er).todense()
        n = adj_mat1.shape[0]
        adj_mat = np.ones((n, n)) - adj_mat1
        m = np.sum(np.triu(adj_mat))

        train, test = DivideNet(adj_mat1, 0.1)
        A_R, A_T = DivideNet(train, 0.1)
        sigma = SPM(train, test, A_R, A_T)
        sigma_c.append(sigma)

        eigvals = np.linalg.eigvals(adj_mat)
        energy = sum(abs(eigvals))

        delta1.append(getMaxEG(n, m) / energy)

    delta2 = [delta1[i] * sigma_c[i] for i in range(4)]
    return sigma_c, delta1, delta2


MLPC_value_arr = np.zeros((10, 4))
sigma_c_arr = np.zeros((10, 4))
delta1_arr = np.zeros((10, 4))
delta2_arr = np.zeros((10, 4))
for repeat in range(10):
    print('repeat=', repeat)
    sigma_c, delta1, delta2 = cal_all_values()
    sigma_c_arr[repeat, :] = sigma_c
    delta1_arr[repeat, :] = delta1
    delta2_arr[repeat, :] = delta2


# 计算表2中ER网络指标值
print(np.mean(delta1_arr, axis=0))
print(np.mean(delta2_arr, axis=0))









