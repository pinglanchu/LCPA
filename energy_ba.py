import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


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
    targets = list(range(m))
    repeated_nodes = []
    source = m
    while source < n:
        G.add_edges_from(zip([source] * m, targets))
        repeated_nodes.extend(targets)
        repeated_nodes.extend([source] * m)
        targets = random_subset(repeated_nodes, m, alpha)
        source += 1

    return G


marker_lst = ['^', 's', 'o', 'd', 'v']

k = 0
for n in [300, 500, 700, 1000]:
    print('当前执行n=%d' % n)
    temp_energy = []
    temp_normalized_energy = []
    for alpha in np.linspace(0, 4, 20):
        BA = get_ba_network(n, 1, alpha)
        adj_mat = nx.adjacency_matrix(BA).todense()
        M = np.sum(np.triu(adj_mat))
        energy_BA = sum(abs(np.linalg.eigvals(adj_mat)))
        temp_energy.append(energy_BA)

    plt.plot(np.linspace(0, 4, 20), temp_energy, marker=marker_lst[k], label='N=%d' % n)
    k += 1
    print('结束当前循环')

plt.xlabel(r'preference degree $\alpha$')
plt.ylabel(r'The energy of scale free network SF($\alpha$)')
plt.legend()
plt.show()


k = 0
for n in [300, 500, 700, 1000]:
    print('当前执行n=%d' % n)
    temp_energy = []
    temp_normalized_energy = []
    for alpha in np.linspace(0, 4, 20):
        BA = get_ba_network(n, 1, alpha)
        adj_mat = nx.adjacency_matrix(BA).todense()
        M = np.sum(np.triu(adj_mat))
        energy_BA = sum(abs(np.linalg.eigvals(adj_mat)))
        temp_energy.append(energy_BA)
        if 2 * M >= n:
            max_EG = 2 * M / n + np.sqrt((n - 1) * (2 * M - (2 * M / n) ** 2))
        else:
            max_EG = np.sqrt(2 * M * n)

        temp_normalized_energy.append(energy_BA / max_EG)
        print('alpha:', alpha, '标准化能量:', max_EG / energy_BA)

    plt.plot(np.linspace(0, 4, 20), temp_normalized_energy, marker=marker_lst[k], label='N=%d' % n)
    k += 1
    print('结束当前循环')

plt.xlabel(r'preference degree $\alpha$')
plt.ylabel('The normalized energy of free scale network')
plt.legend()
plt.show()
