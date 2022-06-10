import scipy.io
import os
from benchmark_algorithm import *
from SPM import SPM
from train_test_split import DivideNet


# (A, q)            'CN' 'RA', 'AA', 'Salton', 'Jaccard', 'PA', 'HPI', 'HDI', 'LHN_I'
# (A, q, lam)       'Katz', 'RWR'
# (A, q, steps)     'LRW', 'SRW'
# (A, q, p)         'SPM'

parameters = [(0.3333, 0.2556), (0.3333, 0.2556), (0.1778, 0.1778), (0.2556, 0.1), (0.3333, 0.1), (0.4889, 0.1),
              (0.1778, 0.3333), (0.4889, 0.1), (0.1778, 0.1778), (0.2556, 0.1778), (0.2556, 0.1778), (0.4111, 0.1)]
files = os.listdir('./Data')

with open('SPM_pH_result.txt', 'w') as f:
    for m in range(12):  # len(files)
        print(files[m])
        data = scipy.io.loadmat('./Data/' + files[m])
        p = parameters[m][1]

        f.write(str(files[m]) + '\n')

        mean_precison_lst = []
        std_lst = []
        q_lst = []

        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            temp_q = 0
            k = 0
            temp_precision = []
            for repeats in range(10):
                try:
                    A = data['A'].todense()
                    A = np.mat(A)
                    train, test = DivideNet(A, q)
                    A_R, A_T = DivideNet(train, p)
                    result = SPM(train, test, A_R, A_T)
                    temp_precision.append(result)
                    print('precision', result)
                    temp_q += q
                    k += 1
                except IndexError as e:
                    pass
                continue
            if k:
                q_lst.append(temp_q / k)
                mean_precison_lst.append(np.mean(temp_precision))
                std_lst.append(np.std(temp_precision))

        f.write('mean' + str(mean_precison_lst) + '\n')
        f.write('std' + str(std_lst) + '\n')
        f.write('q' + str(q_lst) + '\n')

print('CN' + '=' * 20)
with open('CN_pH_result.txt', 'w') as f:
    for m in range(len(files)):
        print(files[m])
        f.write(str(files[m]) + '\n')
        mean_precison_lst = []
        std_lst = []
        q_lst = []
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:  # np.linspace(0.1, 0.8, 8)
            temp_q = 0
            k = 0
            temp_precision = []
            for repeats in range(10):
                try:
                    data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
                    A = data['A'].todense()
                    A = np.mat(A)
                    result = CN(A, q)
                    print('q', q, 'precision', result)
                    temp_precision.append(result)
                    temp_q += q
                    k += 1
                except IndexError as e:
                    pass
                continue
            if k:
                q_lst.append(temp_q / k)
                mean_precison_lst.append(np.mean(temp_precision))
                std_lst.append(np.std(temp_precision))

        f.write('mean' + str(mean_precison_lst) + '\n')
        f.write('std' + str(std_lst) + '\n')
        f.write('q' + str(q_lst) + '\n')
#
print('RA' + '=' * 20)
with open('RA_pH_result.txt', 'w') as f:
    for m in range(len(files)):
        print(files[m])
        f.write(str(files[m]) + '\n')
        mean_precison_lst = []
        std_lst = []
        q_lst = []
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            temp_q = 0
            k = 0
            temp_precision = []
            for repeats in range(10):
                try:
                    data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
                    A = data['A'].todense()
                    A = np.mat(A)
                    result = RA(A, q)
                    print('q', q, 'precision', result)
                    temp_precision.append(result)
                    temp_q += q
                    k += 1
                except IndexError as e:
                    pass
                continue
            if k:
                q_lst.append(temp_q / k)
                mean_precison_lst.append(np.mean(temp_precision))
                std_lst.append(np.std(temp_precision))

        f.write('mean' + str(mean_precison_lst) + '\n')
        f.write('std' + str(std_lst) + '\n')
        f.write('q' + str(q_lst) + '\n')
#
print('AA' + '=' * 20)
with open('AA_pH_result.txt', 'w') as f:
    for m in range(len(files)):
        print(files[m])
        f.write(str(files[m]) + '\n')
        mean_precison_lst = []
        std_lst = []
        q_lst = []
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            temp_q = 0
            k = 0
            temp_precision = []
            for repeats in range(10):
                try:
                    data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
                    A = data['A'].todense()
                    A = np.mat(A)
                    result = AA(A, q)
                    print('q', q, 'precision', result)
                    temp_precision.append(result)
                    temp_q += q
                    k += 1
                except IndexError as e:
                    pass
                continue
            if k:
                q_lst.append(temp_q / k)
                mean_precison_lst.append(np.mean(temp_precision))
                std_lst.append(np.std(temp_precision))

        f.write('mean' + str(mean_precison_lst) + '\n')
        f.write('std' + str(std_lst) + '\n')
        f.write('q' + str(q_lst) + '\n')
#
# print('Salton' + '=' * 20)
# for m in range(len(files)):
#     print(files[m])
#     for q in np.linspace(0.1, 0.8, 8):
#         try:
#             data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#             A = data['A'].todense()
#             A = np.mat(A)
#             print('q', q, 'precision', Salton(A, q))
#         except IndexError as e:
#             pass
#         continue
#
# print('Jarccard' + '=' * 20)
# for m in range(len(files)):
#     print(files[m])
#     for q in np.linspace(0.1, 0.8, 8):
#         try:
#             data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#             A = data['A'].todense()
#             A = np.mat(A)
#             print('q', q, 'precision', Jaccard(A, q))
#         except IndexError as e:
#             pass
#         continue
#
# print('PA' + '=' * 20)
# for m in range(len(files)):
#     print(files[m])
#     for q in np.linspace(0.1, 0.8, 8):
#         try:
#             data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#             A = data['A'].todense()
#             A = np.mat(A)
#             print('q', q, 'precision', PA(A, q))
#         except IndexError as e:
#             pass
#         continue
#
# print('HPI' + '=' * 20)
# for m in range(len(files)):
#     print(files[m])
#     for q in np.linspace(0.1, 0.8, 8):
#         try:
#             data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#             A = data['A'].todense()
#             A = np.mat(A)
#             print('q', q, 'precision', HPI(A, q))
#         except IndexError as e:
#             pass
#         continue
#
# print('HDI' + '=' * 20)
# for m in range(len(files)):
#     print(files[m])
#     for q in np.linspace(0.1, 0.8, 8):
#         try:
#             data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#             A = data['A'].todense()
#             A = np.mat(A)
#             print('q', q, 'precision', HDI(A, q))
#         except IndexError as e:
#             pass
#         continue
#
# print('LHN_I' + '=' * 20)
# for m in range(len(files)):
#     print(files[m])
#     for q in np.linspace(0.1, 0.8, 8):
#         try:
#             data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#             A = data['A'].todense()
#             A = np.mat(A)
#             print('q', q, 'precision', LHN_I(A, q))
#         except IndexError as e:
#             pass
#         continue


# print('Katz' + '=' * 20)
# for m in range(len(files)):
#     print(files[m])
#     for q in np.linspace(0.1, 0.8, 8):
#         for lam in np.linspace(0.1, 0.5, 5):
#             try:
#                 data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#                 A = data['A'].todense()
#                 A = np.mat(A)
#                 print('q', q, 'lam', lam, 'precision', Katz(A, q, lam))
#             except Exception as e:
#                 pass
#             continue

# 计算Katz精度随移除比比例的变化
# with open('Katz_pH_result.txt', 'w') as f:
#     for m in range(len(files)):
#         print(files[m])
#         f.write(str(files[m]))
#         mean_precison_lst = []
#         std_lst = []
#         q_lst = []
#         lam = 0.2
#         for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
#             temp_q = 0
#             k = 0
#             temp_precision = []
#             for repeats in range(10):
#                 try:
#                     data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#                     A = data['A'].todense()
#                     A = np.mat(A)
#                     result = Katz(A, q, lam)
#                     print('q', q, 'lam', lam, 'precision', result)
#                     temp_precision.append(result)
#                     temp_q += q
#                     k += 1
#                 except Exception as e:
#                     pass
#                 continue
#             if k:
#                 q_lst.append(temp_q / k)
#                 mean_precison_lst.append(np.mean(temp_precision))
#                 std_lst.append(np.std(temp_precision))
#
#         f.write('mean' + str(mean_precison_lst) + '\n')
#         f.write('std' + str(std_lst) + '\n')
#         f.write('q' + str(q_lst) + '\n')

#
#
# print('RWR' + '=' * 20)
# for m in range(len(files)):
#     print(files[m])
#     for q in np.linspace(0.1, 0.8, 8):
#         for lam in np.linspace(0.1, 0.5, 5):
#             try:
#                 data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#                 A = data['A'].todense()
#                 A = np.mat(A)
#                 print('q', q, 'lam', lam, 'precision', RWR(A, q, lam))
#             except Exception as e:
#                 pass
#             continue


# print('LRW' + '=' * 20)
# for m in range(len(files)):
#     print(files[m])
#     for q in np.linspace(0.1, 0.8, 8):
#         for steps in [100, 200, 300, 500]:
#             try:
#                 data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#                 A = data['A'].todense()
#                 A = np.mat(A)
#                 print('q', q, 'steps', steps, 'precision', LRW(A, q, steps))
#             except IndexError as e:
#                 pass
#             continue
#
#
# print('SRW' + '=' * 20)
# for m in range(len(files)):
#     print(files[m])
#     for q in np.linspace(0.1, 0.8, 8):
#         for steps in [100, 200, 300, 500]:
#             try:
#                 data = scipy.io.loadmat('Code and Data for NC paper/Data/' + files[m])
#                 A = data['A'].todense()
#                 A = np.mat(A)
#                 print('q', q, 'steps', steps, 'precision', SRW(A, q, steps))
#             except IndexError as e:
#                 pass
#             continue
