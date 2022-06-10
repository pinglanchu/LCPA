import scipy.io
import os
import numpy as np
import networkx as nx


files = os.listdir('./Data')
for k in range(len(files)):
    print(files[k])
    data = scipy.io.loadmat('./Data/' + files[k])
    A = data['A'].todense()
    A = np.mat(A)
    n = A.shape[0]
    m = np.sum(np.triu(A))
    print('节点数：', n)
    print('边数：', m)
    G = nx.from_numpy_matrix(A)
    print('平均聚集系数：', nx.average_clustering(G))
    print('同配系数：', nx.degree_assortativity_coefficient(G))
    eigvals = np.linalg.eigvals(A)

    if 2*m >= n:
        max_EG = 2*m / n + np.sqrt((n-1)*(2*m-(2*m/n)**2))
    else:
        max_EG = np.sqrt(2*m*n)
    print('标准化网络能量：', np.sum(np.abs(eigvals)) / max_EG)
    print('网络密度：', 2 * m / (n * (n-1)))
