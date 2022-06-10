import numpy as np
import networkx as nx
from collections import Counter
import random
import copy
from train_test_split import DivideNet
import os
import scipy.io


def cal_sigma_c(train, test):
    n = train.shape[0]

    L = int(np.sum(np.triu(test)))

    AR_eigvals, AR_eigvecs = np.linalg.eig(train)

    delta_eigvals = []
    appro_eigvecs = []

    # print('first loop')
    for i in range(n):
        temp = AR_eigvecs[:, i].T * test * AR_eigvecs[:, i] / (AR_eigvecs[:, i].T * AR_eigvecs[:, i])
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


def random_subset(repeated_nodes, m, alpha):
    target = set()
    freq = Counter(repeated_nodes)
    weight_degree = sum(value**alpha for value in freq.values())

    for i in range(m):
        p = [(value**alpha) / weight_degree for value in freq.values()]
        node = np.random.choice(list(freq.keys()), p=p)
        target.add(node)
        return target


def get_ba_network(n, m, alpha):
    G = nx.empty_graph(m)
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = random_subset(repeated_nodes, m, alpha)
        source += 1
    return G


# files = os.listdir('./Data')
# for id in range(12):
#     print(id, files[id])
#     data = scipy.io.loadmat('./Data/' + files[id])
#     A = data['A'].todense()
#     train, test = DivideNet(A, 0.1)
#     sigmac = cal_sigma_c(train, test)
#     print(sigmac)

# if __name__ == '__main__':
#     for p in np.linspace(0.1, 0.9, 9):
#         G = nx.erdos_renyi_graph(300, 0.8)  # ER 网络
#         adj_mat = nx.adjacency_matrix(G).todense()
#
#         all_index = []
#         for i in range(300):
#             for j in range(i+1, 300):
#                 if adj_mat[i, j] == 1:
#                     all_index.append((i, j))
#
#         mean_res = 0
#         for repeat in range(10):
#             sampled_index = random.sample(all_index, int(p * len(all_index)))
#             train = copy.deepcopy(adj_mat)
#             for item in sampled_index:
#                 train[item[0], item[1]] = 0
#                 train[item[1], item[0]] = 0
#
#             test = adj_mat - train
#             res = cal_sigma_c(train, test)
#             mean_res += res
#         print(mean_res / 10)

