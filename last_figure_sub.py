import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import networkx as nx
from LCPA import method
from SPM import SPM
from train_test_split import DivideNet
from ws import get_delta
from nw import nw_get_delta


SPCCM = [0.1895, 0.1673, 0.3769, 0.5386, 0.6159, 0.4771, 0.1408, 0.3311, 0.4401, 0.1136, 0.2863, 0.4499]
SPM_value = [0.1911, 0.1586, 0.3874, 0.5334, 0.5842, 0.4687, 0.1347, 0.2974, 0.3911, 0.1081, 0.2542, 0.4002]
NE = [0.5742, 0.7018, 0.5662, 0.5954, 0.4473, 0.6191, 0.6405, 0.5528, 0.5261, 0.6764, 0.5875, 0.5216]
C = [0.6052, 0.2924, 0.2557, 0.0613, 0.2493, 0.4757, 0.6465, 0.3197, 0.2844, 0.1528, 0.2464, 0.6252]

new1 = [1 / NE[i] for i in range(12)]
new2 = [1 / NE[i] * SPM_value[i] for i in range(12)]


def f1(x, a, b):
    return a * x + b


def getMaxEG(n, m):
    if 2 * m >= n:
        max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
    else:
        max_EG = np.sqrt(2*m*n)
    return max_EG


fig0, ax0 = plt.subplots(1, 4, figsize=(15, 4))

a1, b1 = optimize.curve_fit(f1, new1, SPCCM)[0]
y1 = a1 * np.array(new1) + b1

a2, b2 = optimize.curve_fit(f1, new2, SPCCM)[0]
y2 = a2 * np.array(new2) + b2

labels = ['Bio-CE-GT/p=0.1', 'Celegans/p=0.2', 'Ecoli/p=0.3', 'econ-mahindas/p=0.4',
          'econ-wm1/p=0.5', 'facebook/p=0.6', 'metabolic/p=0.7',
          'Political blogs/p=0.8', 'PPI/p=0.9', 'soc-wiki-Vote', 'Tech-routers', 'USAir']
markers = ['^', 'P', 'X', 'v', 'p', '+', 'o', 's', '*', 'x', 'd', '>']

for i in range(12):
    ax0[0].scatter(new1[i], SPCCM[i], label=labels[i], marker=markers[i], s=70)
ax0[0].plot(new1, y1, c='black')
plt.text(0.4, 0.9, 'Real networks', ha='center', va='center', transform=ax0[0].transAxes)
ax0[0].set_xlabel(r'$\delta_{se}$')
ax0[0].set_ylabel('predictability')


markers = ['^', 'P', 'X', 'v', 'p', '+', 'o', 's', '*']
# ########################### er
SPCCM_300 = [0.0809, 0.1628, 0.2576, 0.3503, 0.4392, 0.5373, 0.6353, 0.7475, 0.8604, 0.9752]
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

a1, b1 = optimize.curve_fit(f1, delta1[:-1], SPCCM_300[:-1])[0]
p_lst = np.linspace(0.1, 0.9, 9)
for i in range(9): # label='p=%.2f' % p_lst[i]
    ax0[1].scatter(delta1[i], SPCCM_300[i], label=labels[i],
                     marker=markers[i], s=70)
ax0[1].plot(delta1[:-1], a1 * np.array(delta1)[:-1] + b1, c='black')
plt.text(0.4, 0.9, 'ER network', ha='center', va='center', transform=ax0[1].transAxes)
ax0[1].set_xlabel(r'$\delta_{se}$')
# ax[0, 1].set_ylabel('predictability')

# ######################### ws
ws_delta1_arr, ws_delta2_arr, ws_MLPC_value_arr = get_delta()
a1, b1 = optimize.curve_fit(f1, np.mean(ws_delta1_arr, axis=0),
                            np.mean(ws_MLPC_value_arr, axis=0))[0]

for i in range(9):
    ax0[2].scatter(np.mean(ws_delta1_arr[:, i]), np.mean(ws_MLPC_value_arr[:, i]),
                     marker=markers[i], label=labels[i], s=70)
ax0[2].plot(np.mean(ws_delta1_arr, axis=0), a1 * np.mean(ws_delta1_arr, axis=0) + b1, c='black')
plt.text(0.4, 0.9, 'WS network', ha='center', va='center', transform=ax0[2].transAxes)
ax0[2].set_xlabel(r'$\delta_{se}$')
# ax[0, 2].set_ylabel('predictability')

# ########################### NW
nw_delta1_arr, nw_delta2_arr, nw_MLPC_value_arr = nw_get_delta()
a1, b1 = optimize.curve_fit(f1, np.mean(nw_delta1_arr, axis=0),
                            np.mean(nw_MLPC_value_arr, axis=0))[0]

for i in range(9):
    ax0[3].scatter(np.mean(nw_delta1_arr[:, i]), np.mean(nw_MLPC_value_arr[:, i]),
                     marker=markers[i], label=labels[i], s=70)
ax0[3].plot(np.mean(nw_delta1_arr, axis=0), a1 * np.mean(nw_delta1_arr, axis=0) + b1, c='black')
plt.text(0.4, 0.9, 'NW network', ha='center', va='center', transform=ax0[3].transAxes)
ax0[3].set_xlabel(r'$\delta_{se}$')
# ax[0, 3].set_ylabel('predictability')
lines, labels = ax0[0].get_legend_handles_labels()
fig0.legend(lines, labels, bbox_to_anchor=(0.122, 0.9, 0.78, 0.2), loc='lower left',
                      ncol=6, mode="expand", borderaxespad=0.)
plt.show()
plt.close()


fig1, ax1 = plt.subplots(1, 4, figsize=(15, 4))
markers = ['^', 'P', 'X', 'v', 'p', '+', 'o', 's', '*', 'x', 'd', '>']
for i in range(12):
    ax1[0].scatter(new2[i], SPCCM[i], label=labels[i], marker=markers[i], s=70)
ax1[0].plot(new2, y2, c='black')
plt.text(0.4, 0.9, 'Real networks', ha='center', va='center', transform=ax1[0].transAxes)
ax1[0].set_xlabel(r'$\delta_{sesc}$')
ax1[0].set_ylabel('predictability')

a2, b2 = optimize.curve_fit(f1, delta2[:-1], SPCCM_300[:-1])[0]
for i in range(9):
    ax1[1].scatter(delta2[i], SPCCM_300[i], label=labels[i], marker=markers[i], s=70)
ax1[1].plot(delta2[:-1], a2 * np.array(delta2)[:-1] + b2, c='black')
plt.text(0.4, 0.9, 'ER network', ha='center', va='center', transform=ax1[1].transAxes)
ax1[1].set_xlabel(r'$\delta_{sesc}$')
# ax[1, 1].set_ylabel('predictability')

a2, b2 = optimize.curve_fit(f1, np.mean(ws_delta2_arr, axis=0),
                            np.mean(ws_MLPC_value_arr, axis=0))[0]
for i in range(9):
    ax1[2].scatter(np.mean(ws_delta2_arr[:, i]), np.mean(ws_MLPC_value_arr[:, i]),
                     marker=markers[i], label=labels[i], s=70)
ax1[2].plot(np.mean(ws_delta2_arr, axis=0), a2 * np.mean(ws_delta2_arr, axis=0) + b2, c='black')
plt.text(0.4, 0.9, 'WS network', ha='center', va='center', transform=ax1[2].transAxes)
ax1[2].set_xlabel(r'$\delta_{sesc}$')

a2, b2 = optimize.curve_fit(f1, np.mean(nw_delta2_arr, axis=0),
                            np.mean(nw_MLPC_value_arr, axis=0))[0]
for i in range(9):
    ax1[3].scatter(np.mean(nw_delta2_arr[:, i]), np.mean(nw_MLPC_value_arr[:, i]),
                     marker=markers[i], label=labels[i], s=70)
ax1[3].plot(np.mean(nw_delta2_arr, axis=0), a2 * np.mean(nw_delta2_arr, axis=0) + b2, c='black')
plt.text(0.4, 0.9, 'NW network', ha='center', va='center', transform=ax1[3].transAxes)
ax1[3].set_xlabel(r'$\delta_{sesc}$')
# ax[1, 3].set_ylabel('predictability')

lines, labels = ax1[0].get_legend_handles_labels()
fig1.legend(lines, labels, bbox_to_anchor=(0.122, 0.9, 0.78, 0.2), loc='lower left',
            ncol=6, mode="expand", borderaxespad=0.)
plt.show()
