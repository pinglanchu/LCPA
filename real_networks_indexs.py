import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import random
from LCPA import method
from SPM import SPM
from train_test_split import DivideNet
from scipy import optimize


def getMaxEG(n, m):
    if 2 * m >= n:
        max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
    else:
        max_EG = np.sqrt(2*m*n)
    return max_EG


def f1(x, a, b):
    return a * x + b


# q, p
parameters = [(0.3333, 0.2556), (0.3333, 0.2556), (0.1778, 0.1778), (0.2556, 0.1),
              (0.3333, 0.1), (0.4889, 0.1), (0.1778, 0.3333), (0.4889, 0.1),
              (0.1778, 0.1778), (0.2556, 0.1778), (0.2556, 0.1778), (0.4111, 0.1)]

fig, ax = plt.subplots(4, 6, figsize=(25, 15))
files = os.listdir('./Data')
for id in range(6, 7):
    print(id, files[id])
    data = scipy.io.loadmat('./Data/' + files[id])
    delta_se = []
    LCPA = []
    sigma_c = []
    for p in np.linspace(0.01, 0.2, 10):
        result = 0
        sigma = 0
        energy = 0
        maxeg = 0
        k = 0
        for repeat in range(5):
            try:
                A = data['A'].todense()
                n = A.shape[0]
                A = np.ones((n, n)) - A
                one_index = np.argwhere(np.triu(A) == 1).tolist()
                sampled = random.sample(one_index, int(p*len(one_index)))
                for m in sampled:
                    A[m[0], m[1]] = 0
                    A[m[1], m[0]] = 0

                m = np.sum(np.triu(A))
                result += method(A, parameters[id][0], parameters[id][1])
                train, test = DivideNet(A, parameters[id][0])
                A_R, A_T = DivideNet(train, parameters[id][1])
                sigma += SPM(train, test, A_R, A_T)
                eigvals = np.linalg.eigvals(A)
                energy += sum(abs(eigvals))
                maxeg += getMaxEG(n, m)
                k += 1
            except IndexError as e:
                pass
            continue
        delta_se.append(maxeg / energy)
        LCPA.append(result / k)
        sigma_c.append(sigma / k)
    delta_sesc = [delta_se[i] * sigma_c[i] for i in range(10)]

    if id < 6:
        ax[0, id].scatter(delta_se, LCPA, marker='s', label=files[id][:-4])
        a, b = optimize.curve_fit(f1, delta_se, LCPA)[0]
        print('a=%.4f, b=%.4f' % (a, b))
        ax[0, id].plot(delta_se, a * np.array(delta_se) + b, c='black')
        plt.text(0.6, 0.8, files[id][:-4], ha='center', va='center', transform=ax[0, id].transAxes)
        ax[0, 0].set_ylabel('LCPA precision')

        ax[2, id].scatter(delta_sesc, LCPA, marker='^', label=files[id][:-4])
        a, b = optimize.curve_fit(f1, delta_sesc, LCPA)[0]
        print('a=%.4f, b=%.4f' % (a, b))
        ax[2, id].plot(delta_sesc, a * np.array(delta_sesc) + b, c='black')
        plt.text(0.4, 0.8, files[id][:-4], ha='center', va='center', transform=ax[2, id].transAxes)
        ax[2, 0].set_ylabel('LCPA precision')
    else:
        ax[1, id-6].scatter(delta_se, LCPA, marker='s', label=files[id][:-4])
        a, b = optimize.curve_fit(f1, delta_se, LCPA)[0]
        print('a=%.4f, b=%.4f' % (a, b))
        ax[1, id-6].plot(delta_se, a * np.array(delta_se) + b, c='black')
        plt.text(0.7, 0.8, files[id][:-4], ha='center', va='center', transform=ax[1, id-6].transAxes)
        ax[1, id - 6].set_xlabel(r'$\delta_{se}$')
        ax[1, 0].set_ylabel('LCPA precision')

        ax[3, id-6].scatter(delta_se, LCPA, marker='^', label=files[id][:-4])
        a, b = optimize.curve_fit(f1, delta_sesc, LCPA)[0]
        print('a=%.4f, b=%.4f' % (a, b))
        ax[3, id-6].plot(delta_sesc, a * np.array(delta_sesc) + b, c='black')
        plt.text(0.4, 0.8, files[id][:-4], ha='center', va='center', transform=ax[3, id-6].transAxes)
        ax[3, id-6].set_xlabel(r'$\delta_{sesc}$')
        ax[3, 0].set_ylabel('LCPA precision')

plt.show()