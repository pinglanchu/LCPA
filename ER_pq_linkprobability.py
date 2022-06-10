import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from itertools import combinations
from LCPA import method


SPCCM_100 = [0.16875, 0.24,  0.3553,  0.4552, 0.5392, 0.6180, 0.7382, 0.8563, 0.975]

# N = 300
SPCCM_300 = [0.0809, 0.1628, 0.2576, 0.3503, 0.4392, 0.5373, 0.6353, 0.7475, 0.8604, 0.9752]

# N = 500
SPCCM_500 = [0.0825, 0.1578, 0.2472, 0.3412, 0.4374, 0.5378, 0.6387, 0.74539, 0.8586, 0.9752]

# N = 700
SPCCM_700 = [0.0823, 0.1674, 0.2545, 0.3425, 0.4418, 0.5395, 0.6416, 0.7481, 0.8597, 0.9756]

plt.figure(figsize=(10, 6))
plt.scatter(np.linspace(0.2, 1, 9), SPCCM_100, marker='o', label='N=100')
plt.scatter(np.linspace(0.1, 1, 10), SPCCM_300, marker='s', label='N=300')
plt.scatter(np.linspace(0.1, 1, 10), SPCCM_500, marker='*', label='N=500')
plt.scatter(np.linspace(0.1, 1, 10), SPCCM_700, marker='+', label='N=700')
plt.plot(np.linspace(0.1, 1, 10), np.linspace(0.1, 1, 10), linestyle='--', color='black', label=r'$\hat{p}=p$')
plt.legend()
plt.xlabel('p')
plt.ylabel(r'$\hat{p}$')
plt.show()







