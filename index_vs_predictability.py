from ws import get_sigmac
from nw import nw_get_sigmac
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
from ws import get_delta
from nw import nw_get_delta
import networkx as nx


def f1(x, a, b):
    return a * x + b


def getMaxEG(n, m):
    if 2 * m >= n:
        max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
    else:
        max_EG = np.sqrt(2*m*n)
    return max_EG


labels = ['Bio-CE-GT', 'Celegans', 'Ecoli', 'econ-mahindas',
          'econ-wm1', 'facebook', 'metabolic',
          'Political blogs', 'PPI', 'soc-wiki-Vote', 'Tech-routers', 'USAir']
markers = ['^', 'P', 'X', 'v', 'p', '+', 'o', 's', '*', 'x', 'd', '>']

fig0, ax0 = plt.subplots(1, 3, figsize=(15, 4))

LCPA = [0.1895, 0.1673, 0.3769, 0.5386, 0.6159, 0.4771, 0.1408, 0.3311, 0.4401, 0.1136, 0.2863, 0.4499]
sigma_c = [0.2105, 0.2570, 0.4273, 0.5752, 0.6933, 0.4144, 0.1138, 0.2489, 0.5241, 0.1203, 0.3032, 0.4764]

a1, b1 = optimize.curve_fit(f1, sigma_c, LCPA)[0]
y1 = a1 * np.array(sigma_c) + b1
for i in range(12):
    ax0[0].scatter(sigma_c[i], LCPA[i], label=labels[i], marker=markers[i], s=70)
ax0[0].plot(sigma_c, y1, c='black')
plt.text(0.4, 0.9, '(a)', ha='center', va='center', transform=ax0[0].transAxes)
ax0[0].set_xlabel(r'$\sigma_c$')
ax0[0].set_ylabel('predictability')

LCPA = [0.1895, 0.1673, 0.3769, 0.5386, 0.6159, 0.4771, 0.1408, 0.3311, 0.4401, 0.1136, 0.2863, 0.4499]
SPM = [0.1911, 0.1586, 0.3874, 0.5334, 0.5842, 0.4687, 0.1347, 0.2974, 0.3911, 0.1081, 0.2542, 0.4002]
NE = [0.5742, 0.7018, 0.5662, 0.5954, 0.4473, 0.6191, 0.6405, 0.5528, 0.5261, 0.6764, 0.5875, 0.5216]
C = [0.6052, 0.2924, 0.2557, 0.0613, 0.2493, 0.4757, 0.6465, 0.3197, 0.2844, 0.1528, 0.2464, 0.6252]

new1 = [1 / NE[i] for i in range(12)]
new2 = [1 / NE[i] * SPM[i] for i in range(12)]

a1, b1 = optimize.curve_fit(f1, new1, LCPA)[0]
y1 = a1 * np.array(new1) + b1
for i in range(12):
    ax0[1].scatter(new1[i], LCPA[i], label=labels[i], marker=markers[i], s=70)
ax0[1].plot(new1, y1, c='black')
plt.text(0.4, 0.9, '(b)', ha='center', va='center', transform=ax0[1].transAxes)
ax0[1].set_xlabel(r'$\delta_{se}$')
ax0[1].set_ylabel('predictability')

a2, b2 = optimize.curve_fit(f1, new2, LCPA)[0]
y2 = a2 * np.array(new2) + b2
for i in range(12):
    ax0[2].scatter(new2[i], LCPA[i], label=labels[i], marker=markers[i], s=70)
ax0[2].plot(new2, y2, c='black')
plt.text(0.4, 0.9, '(c)', ha='center', va='center', transform=ax0[2].transAxes)
ax0[2].set_xlabel(r'$\delta_{sesc}$')
ax0[2].set_ylabel('predictability')
lines, labels = ax0[0].get_legend_handles_labels()
fig0.legend(lines, labels, bbox_to_anchor=(0.122, 0.9, 0.78, 0.2), loc='lower left',
                      ncol=6, mode="expand", borderaxespad=0.)
plt.show()


# ########################### ER
fig1, ax1 = plt.subplots(1, 3, figsize=(15, 4))
labels = ['p=0.1', 'p=0.2', 'p=0.3', 'p=0.4', 'p=0.5', 'p=0.6', 'p=0.7', 'p=0.8', 'p=0.9']
LCPA_300 = [0.0809, 0.1628, 0.2576, 0.3503, 0.4392, 0.5373, 0.6353, 0.7475, 0.8604, 0.9752]
sigma_c = [0.3540, 0.4876, 0.5707, 0.6304, 0.6766, 0.7135, 0.7412, 0.7651, 0.7836, 0.7985]

a1, b1 = optimize.curve_fit(f1, sigma_c[-1], LCPA_300[:-1])[0]
for i in range(9):
    ax1[0].scatter(sigma_c[i], LCPA_300[i], label=labels[i], marker=markers[i], s=70)
ax1[0].plot(sigma_c[:-1], a1 * np.array(sigma_c)[:-1] + b1, c='black')
plt.text(0.4, 0.9, '(a)', ha='center', va='center', transform=ax1[0].transAxes)
ax1[0].set_xlabel(r'$\sigma_c$')

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

a1, b1 = optimize.curve_fit(f1, delta1[:-1], LCPA_300[:-1])[0]
p_lst = np.linspace(0.1, 0.9, 9)
for i in range(9):
    ax1[1].scatter(delta1[i], LCPA_300[i], label=labels[i], marker=markers[i], s=70)
ax1[1].plot(delta1[:-1], a1 * np.array(delta1)[:-1] + b1, c='black')
plt.text(0.4, 0.9, '(b)', ha='center', va='center', transform=ax1[1].transAxes)
ax1[1].set_xlabel(r'$\delta_{se}$')

a2, b2 = optimize.curve_fit(f1, delta2[:-1], LCPA_300[:-1])[0]
for i in range(9):
    ax1[2].scatter(delta2[i], LCPA_300[i], label=labels[i], marker=markers[i], s=70)
ax1[2].plot(delta2[:-1], a2 * np.array(delta2)[:-1] + b2, c='black')
plt.text(0.4, 0.9, '(c)', ha='center', va='center', transform=ax1[2].transAxes)
ax1[2].set_xlabel(r'$\delta_{sesc}$')
lines, labels = ax1[0].get_legend_handles_labels()
fig1.legend(lines, labels, bbox_to_anchor=(0.122, 0.9, 0.78, 0.2), loc='lower left',
            ncol=9, mode="expand", borderaxespad=0.)
plt.show()


# #############################WS
fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
labels = ['p=0.1', 'p=0.2', 'p=0.3', 'p=0.4', 'p=0.5', 'p=0.6', 'p=0.7', 'p=0.8', 'p=0.9']
sigma_c_arr, ws_LCPA_value_arr = get_sigmac()

a1, b1 = optimize.curve_fit(f1, np.mean(sigma_c_arr, axis=0),
                            np.mean(ws_LCPA_value_arr, axis=0))[0]
for i in range(9):
    ax2[0].scatter(np.mean(sigma_c_arr[:, i]), np.mean(ws_LCPA_value_arr[:, i]),
                   marker=markers[i], label=labels[i], s=70)
ax2[0].plot(np.mean(sigma_c_arr, axis=0), a1 * np.mean(sigma_c_arr, axis=0) + b1, c='black')
plt.text(0.4, 0.9, '(a)', ha='center', va='center', transform=ax2[0].transAxes)
ax2[0].set_xlabel(r'$\sigma_c$')


ws_delta1_arr, ws_delta2_arr, ws_LCPA_value_arr = get_delta()
a1, b1 = optimize.curve_fit(f1, np.mean(ws_delta1_arr, axis=0),
                            np.mean(ws_LCPA_value_arr, axis=0))[0]
for i in range(9):
    ax2[1].scatter(np.mean(ws_delta1_arr[:, i]), np.mean(ws_LCPA_value_arr[:, i]),
                   marker=markers[i], label=labels[i], s=70)
ax2[1].plot(np.mean(ws_delta1_arr, axis=0), a1 * np.mean(ws_delta1_arr, axis=0) + b1, c='black')
plt.text(0.4, 0.9, '(b)', ha='center', va='center', transform=ax2[1].transAxes)
ax2[1].set_xlabel(r'$\delta_{se}$')

a2, b2 = optimize.curve_fit(f1, np.mean(ws_delta2_arr, axis=0),
                            np.mean(ws_LCPA_value_arr, axis=0))[0]
for i in range(9):
    ax2[2].scatter(np.mean(ws_delta2_arr[:, i]), np.mean(ws_LCPA_value_arr[:, i]),
                   marker=markers[i], label=labels[i], s=70)
ax2[2].plot(np.mean(ws_delta2_arr, axis=0), a2 * np.mean(ws_delta2_arr, axis=0) + b2, c='black')
plt.text(0.4, 0.9, '(c)', ha='center', va='center', transform=ax2[2].transAxes)
ax2[2].set_xlabel(r'$\delta_{sesc}$')
lines, labels = ax2[0].get_legend_handles_labels()
fig2.legend(lines, labels, bbox_to_anchor=(0.122, 0.9, 0.78, 0.2), loc='lower left',
            ncol=9, mode="expand", borderaxespad=0.)
plt.show()


