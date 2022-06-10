import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

marker = ['^', 's', 'o', 'd']
k = 0
for n in [300, 500, 700, 1000]:
    print('当前执行n=%d' % n)
    energy_lst = []
    normalized_energy_lst = []
    for p in np.linspace(0, 1, 10):
        ER = nx.erdos_renyi_graph(n, p)
        adj_mat = nx.adjacency_matrix(ER).todense()
        eigvals = np.linalg.eigvals(adj_mat)
        energy_ER = sum(abs(eigvals))
        energy_lst.append(energy_ER)

        adj_mat = np.ones((n, n)) - adj_mat
        M = np.sum(np.triu(adj_mat))
        temp_energy = sum(abs(np.linalg.eigvals(adj_mat)))
        if 2 * M >= n:
            max_EG = 2 * M / n + np.sqrt((n - 1) * (2 * M - (2 * M / n) ** 2))
        else:
            max_EG = np.sqrt(2 * M * n)
        normalized_energy_lst.append(temp_energy / max_EG)

    axes[0].plot(np.linspace(0.1, 1, 10), energy_lst, marker=marker[k], label='N=%d' % n)
    axes[1].plot(np.linspace(0.1, 1, 10), normalized_energy_lst, marker=marker[k], label='N=%d' % n)
    k += 1

axes[0].set_xlabel(r'link probability $p$')
axes[0].set_ylabel('The non-normalized energy of network ER(p)')
axes[1].set_xlabel(r'link probability $p$')
axes[1].set_ylabel('The normalized energy of network ER(p)')
plt.show()

