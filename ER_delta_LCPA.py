import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import networkx as nx


def getMaxEG(n, m):
    if 2 * m >= n:
        max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
    else:
        max_EG = np.sqrt(2*m*n)
    return max_EG


def f1(x, a, b):
    return a * x + b


LCPA_300 = [0.0809, 0.1628, 0.2576, 0.3503, 0.4392, 0.5373, 0.6353, 0.7475, 0.8604, 0.9752]
sigma_c = [0.3540, 0.4876, 0.5707, 0.6304, 0.6766, 0.7135, 0.7412, 0.7651, 0.7836, 0.7985]

delta1 = []
for p in np.linspace(0.1, 1, 10):
    G = nx.erdos_renyi_graph(300, p)
    adj_mat = nx.adjacency_matrix(G).todense()
    adj_mat = np.ones((300, 300)) - adj_mat
    n = 300
    m = np.sum(np.triu(adj_mat))
    eigvals = np.linalg.eigvals(adj_mat)
    non_energy = sum(abs(eigvals))
    delta1.append(getMaxEG(n, m) / non_energy)


delta2 = [sigma_c[i] * delta1[i] for i in range(10)]

a, b = optimize.curve_fit(f1, delta1[:-1], LCPA_300[:-1])[0]
print('a', a, 'b', b)

markers = ['^', 'P', 'X', 'v', 'p', '+', 'o', 's', '*', 'x']
p_lst = np.linspace(0.1, 1, 10)
for i in range(9):
    plt.scatter(delta1[i], LCPA_300[i], label='p=%.2f' % p_lst[i], marker=markers[i], s=70)
plt.plot(delta1[:-1], a * np.array(delta1)[:-1] + b, c='black')
plt.legend(loc='upper left', frameon=False)
plt.xlabel(r'$\delta_{se}$')
plt.ylabel('predictability')
plt.show()




