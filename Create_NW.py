import networkx as nx
import numpy as np
import random


def get_nw(n, k, p):
    G = nx.random_regular_graph(k, n)
    A = nx.adjacency_matrix(G).todense()
    index = np.argwhere(A == 0)

    for item in index:
        if item[0] != item[1]:
            if random.random() < p:
                A[item[0], item[1]] = 1

    return A
