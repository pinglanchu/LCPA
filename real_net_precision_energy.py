import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from train_test_split import DivideNet
from scipy import optimize


# files = os.listdir('./Data')
#
# q_lst = [0.2556, 0.4889, 0.1, 0.1, 0.1, 0.3333, 0.2556, 0.6444, 0.1778,	0.4111,	0.1778, 0.2556]
# p_lst = [0.1, 0.1, 0.2556, 0.1, 0.1, 0.1, 0.1778, 0.1, 0.1778, 0.1, 0.2556, 0.1778]
#
# Epsilon = []
# k = 0
# for file in files:
#     print(file)
#     data = scipy.io.loadmat('./Data/' + file)
#     # 初始完整网络
#     adj_mat = data['A'].todense()
#     n = adj_mat.shape[0]
#     m = np.sum(adj_mat) / 2
#     eign_real = np.linalg.eigvals(adj_mat)
#     energy_real = sum(abs(eign_real))
#
#     if 2*m >= n:
#         max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
#     else:
#         max_EG = np.sqrt(2*m*n)
#     normalized_real = energy_real / max_EG
#
#     # 观测网络
#     train, _ = DivideNet(adj_mat, q_lst[k])
#     m = np.sum(train) / 2
#     eign_train = np.linalg.eigvals(train)
#     energy_train = sum(abs(eign_train))
#     if 2*m >= n:
#         max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
#     else:
#         max_EG = np.sqrt(2*m*n)
#     normalized_train = energy_train / max_EG
#
#     # 摄动之后网络
#     AR, _ = DivideNet(train, p_lst[k])
#     eign_AR = np.linalg.eigvals(AR)
#     energy_AR = sum(abs(eign_AR))
#     m = np.sum(AR) / 2
#     if 2*m >= n:
#         max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
#     else:
#         max_EG = np.sqrt(2*m*n)
#     normalized_AR = energy_AR / max_EG
#
#     # 摄动前后网络标准化能量比值
#     Epsilon.append(normalized_train / normalized_real)
#     k += 1

LCPA = [0.1895, 0.1673, 0.3769, 0.5386, 0.6159, 0.4771, 0.1408, 0.3311, 0.4401, 0.1136, 0.2863, 0.4499]
SPM = [0.1911, 0.1586, 0.3874, 0.5334, 0.5842, 0.4687, 0.1347, 0.2974, 0.3911, 0.1081, 0.2542, 0.4002]
NE = [0.5742, 0.7018, 0.5662, 0.5954, 0.4473, 0.6191, 0.6405, 0.5528, 0.5261, 0.6764, 0.5875, 0.5216]
C = [0.6052, 0.2924, 0.2557, 0.0613, 0.2493, 0.4757, 0.6465, 0.3197, 0.2844, 0.1528, 0.2464, 0.6252]

new1 = [1 / NE[i] for i in range(12)]
new2 = [1 / NE[i] * SPM[i] for i in range(12)]


def f1(x, a, b):
    return a * x + b


a, b = optimize.curve_fit(f1, new1, LCPA)[0]

print('a', a, 'b', b)
y1 = a * np.array(new1) + b

labels = ['Bio-CE-GT', 'Celegans', 'Ecoli', 'econ-mahindas', 'econ-wm1', 'facebook', 'metabolic',
          'Political blogs', 'PPI', 'soc-wiki-Vote', 'Tech-routers', 'USAir']
markers = ['^', 'P', 'X', 'v', 'p', '+', 'o', 's', 'd', '>', '*', 'x']
for i in range(12):
    plt.scatter(new1[i], LCPA[i], label=labels[i], marker=markers[i], s=70)
plt.plot(new1, y1, c='black')
plt.legend(loc='upper left', frameon=False)
plt.xlabel(r'$\delta_{se}$')
plt.ylabel('predictability')
plt.show()

# print(new1)
# print(new2)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1 = ax.plot(SPCCM, marker='s', linestyle='--', color='red', label='LCPA')
# ax.set_ylabel('precision')
#
#
# ax_twin = ax.twinx()
# line2 = ax_twin.plot(new2, marker='^', linestyle='--', color='green', label=r'$\delta_{sesc}$')
#
# ax_twin.set_ylabel(r'$\delta_{sesc}$ value')
#
#
# plt.xticks(range(12), ['Bio-CE-GT', 'C.elegans', 'E.coli', 'econ-mahindas', 'econ-wm1',
#                        'Facebook', 'Metabolic', 'Political blog', 'PPI', 'Soc-wike-vote',
#                        'Tech-routers', 'USAir'])
#
# for tl in ax.get_xticklabels():
#     tl.set_rotation(-45)
#     tl.set_fontsize(8)
#
# lns = line1 + line2
# ax.legend(lns, [l.get_label() for l in lns])
#
# plt.savefig('能量部分/delta_sesc.png', dpi=600, bbox_inches='tight')
#
# plt.show()
