import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io
import os
import random

fig, ax = plt.subplots(2, 4, figsize=(15, 6))
E = []
SVE = []
for p in np.arange(0.05, 0.55, 0.05):
    sum_eigval_A = 0
    sve = 0
    for i in range(10):
        G = nx.erdos_renyi_graph(100, p)
        adj_mat0 = nx.adjacency_matrix(G).todense()
        n = adj_mat0.shape[0]
        adj_mat = adj_mat0
        Sigma = 1 / (n-1) * np.dot(adj_mat, adj_mat.T)
        eigval = np.linalg.eigvals(Sigma)
        sum_eigval = np.sum(eigval)
        eigval_A = np.linalg.eigvals(adj_mat)
        sum_eigval_A += sum(abs(eigval_A))
        sve += -np.sum(eigval / sum_eigval * np.log2(eigval / sum_eigval+1e-10))
    E.append(sum_eigval_A/10)
    SVE.append(sve/10)
ax[0, 0].plot(E, SVE, marker='s')
ax[0, 0].set_ylabel(r'$H_{sv}$')
plt.text(0.6, 0.8, 'ER', ha='center', va='center', transform=ax[0, 0].transAxes)

E = []
SVE = []
for p in np.arange(0.05, 0.55, 0.05):
    sum_eigval_A = 0
    sve = 0
    times = 0
    for i in range(10):
        G = nx.newman_watts_strogatz_graph(100, 3, p)
        adj_mat0 = nx.adjacency_matrix(G).todense()
        n = adj_mat0.shape[0]
        Sigma = 1 / (n-1) * np.dot(adj_mat0, adj_mat0.T)
        eigval = np.linalg.eigvals(Sigma)
        sum_eigval = sum(eigval)
        eigval_A = np.linalg.eigvals(adj_mat0)
        sum_eigval_A += sum(abs(eigval_A))
        sve += -np.sum(eigval / sum_eigval * np.log2(eigval / sum_eigval+1e-10))
        times += 1
    E.append(sum_eigval_A/times)
    SVE.append(np.round(sve/times, 2))
ax[0, 1].scatter(E, SVE, marker='s')
ax[0, 1].plot(E, SVE)
plt.text(0.6, 0.8, 'NW', ha='center', va='center', transform=ax[0, 1].transAxes)


files = os.listdir('Code and Data for NC paper/Data')
id_ = 0
for id in [0, 1, 5, 7, 10, 11]:
    print(id, files[id])
    data = scipy.io.loadmat('/Data/' + files[id])
    E1 = []
    SVE1 = []
    for p in np.arange(0.05, 0.55, 0.05):
        sum_eigval = 0
        sum_eigval_A = 0
        entropy = 0
        for k in range(10):
            A = data['A'].todense()
            one_index = np.argwhere(np.triu(A) == 1).tolist()
            sampled = random.sample(one_index, int(p*len(one_index)))
            for m in sampled:
                A[m[0], m[1]] = 0
                A[m[1], m[0]] = 0
            Sigma = 1 / (len(A) - 1) * np.dot(A, A.T)
            eigval_A = np.linalg.eigvals(A)
            eigval = np.linalg.eigvals(Sigma)
            eigval[eigval < 1e-10] = 0
            eigval = list(set(np.real(eigval).tolist()))
            eigval.remove(0)
            sum_eigval_A += np.sum(abs(eigval_A))
            sum_eigval += np.sum(eigval)
            entropy += -np.sum(eigval / sum_eigval * np.log2(eigval / sum_eigval))
        SVE1.append(entropy/10)
        E1.append(sum_eigval_A/10)
    if id_ < 2:
        ax[0, id_+2].plot(E1, SVE1, marker='s', label=files[id][:-4])
        plt.text(0.6, 0.8, files[id][:-4], ha='center', va='center', transform=ax[0, id_+2].transAxes)
    elif id_ == 2:
        ax[1, id_-2].plot(E1, SVE1, marker='s', label=files[id][:-4])
        plt.text(0.6, 0.8, files[id][:-4], ha='center', va='center', transform=ax[1, id_-2].transAxes)
        ax[1, id_-2].set_xlabel(r'$E(G)$')
        ax[1, id_-2].set_ylabel(r'$H_{sv}$')
    else:
        ax[1, id_-2].plot(E1, SVE1, marker='s', label=files[id][:-4])
        plt.text(0.6, 0.8, files[id][:-4], ha='center', va='center', transform=ax[1, id_-2].transAxes)
        ax[1, id_-2].set_xlabel(r'$E(G)$')

    id_ += 1

plt.show()
