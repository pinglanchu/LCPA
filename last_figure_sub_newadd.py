from ws import get_sigmac
from nw import nw_get_sigmac
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np


def f1(x, a, b):
    return a * x + b


def getMaxEG(n, m):
    if 2 * m >= n:
        max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
    else:
        max_EG = np.sqrt(2*m*n)
    return max_EG


labels = ['Bio-CE-GT/p=0.1', 'Celegans/p=0.2', 'Ecoli/p=0.3', 'econ-mahindas/p=0.4',
          'econ-wm1/p=0.5', 'facebook/p=0.6', 'metabolic/p=0.7',
          'Political blogs/p=0.8', 'PPI/p=0.9', 'soc-wiki-Vote', 'Tech-routers', 'USAir']
markers = ['^', 'P', 'X', 'v', 'p', '+', 'o', 's', '*', 'x', 'd', '>']

fig2, ax2 = plt.subplots(1, 4, figsize=(15, 4))

SPCCM = [0.1895, 0.1673, 0.3769, 0.5386, 0.6159, 0.4771, 0.1408, 0.3311, 0.4401, 0.1136, 0.2863, 0.4499]
sigma_c = [0.2105, 0.2570, 0.4273, 0.5752, 0.6933, 0.4144, 0.1138, 0.2489, 0.5241, 0.1203, 0.3032, 0.4764]
a1, b1 = optimize.curve_fit(f1, sigma_c, SPCCM)[0]
y1 = a1 * np.array(sigma_c) + b1

for i in range(12):
    ax2[0].scatter(sigma_c[i], SPCCM[i], label=labels[i], marker=markers[i], s=70)
ax2[0].plot(sigma_c, y1, c='black')
plt.text(0.4, 0.9, 'Real networks', ha='center', va='center', transform=ax2[0].transAxes)
ax2[0].set_xlabel(r'$\sigma_c$')
ax2[0].set_ylabel('predictability')

SPCCM_300 = [0.0809, 0.1628, 0.2576, 0.3503, 0.4392, 0.5373, 0.6353, 0.7475, 0.8604, 0.9752]
sigma_c = [0.3540, 0.4876, 0.5707, 0.6304, 0.6766, 0.7135, 0.7412, 0.7651, 0.7836, 0.7985]
a1, b1 = optimize.curve_fit(f1, sigma_c[-1], SPCCM_300[:-1])[0]
for i in range(9):
    ax2[1].scatter(sigma_c[i], SPCCM_300[i], label=labels[i], marker=markers[i], s=70)
ax2[1].plot(sigma_c[:-1], a1 * np.array(sigma_c)[:-1] + b1, c='black')
plt.text(0.4, 0.9, 'ER network', ha='center', va='center', transform=ax2[1].transAxes)
ax2[1].set_xlabel(r'$\sigma_c$')


sigma_c_arr, ws_MLPC_value_arr = get_sigmac()

a1, b1 = optimize.curve_fit(f1, np.mean(sigma_c_arr, axis=0),
                            np.mean(ws_MLPC_value_arr, axis=0))[0]

for i in range(9):
    ax2[2].scatter(np.mean(sigma_c_arr[:, i]), np.mean(ws_MLPC_value_arr[:, i]),
                     marker=markers[i], label=labels[i], s=70)
ax2[2].plot(np.mean(sigma_c_arr, axis=0), a1 * np.mean(sigma_c_arr, axis=0) + b1, c='black')
plt.text(0.4, 0.9, 'WS network', ha='center', va='center', transform=ax2[2].transAxes)
ax2[2].set_xlabel(r'$\sigma_c$')

sigma_c_arr, nw_MLPC_value_arr = nw_get_sigmac()
a1, b1 = optimize.curve_fit(f1, np.mean(sigma_c_arr, axis=0),
                            np.mean(nw_MLPC_value_arr, axis=0))[0]

for i in range(9):
    ax2[3].scatter(np.mean(sigma_c_arr[:, i]), np.mean(nw_MLPC_value_arr[:, i]),
                     marker=markers[i], label=labels[i], s=70)
ax2[3].plot(np.mean(sigma_c_arr, axis=0), a1 * np.mean(sigma_c_arr, axis=0) + b1, c='black')
plt.text(0.4, 0.9, 'NW network', ha='center', va='center', transform=ax2[3].transAxes)
ax2[3].set_xlabel(r'$\sigma_c$')

lines, labels = ax2[0].get_legend_handles_labels()
fig2.legend(lines, labels, bbox_to_anchor=(0.122, 0.9, 0.78, 0.2), loc='lower left',
            ncol=6, mode="expand", borderaxespad=0.)
plt.show()
