import numpy as np
from scipy import sparse


def spones(array):
    """
    将矩阵中的非零元素全部替换为 1
    """
    sparse_array = sparse.csr_matrix(array)  # 转换为稀疏矩阵
    sparse_array.data.fill(1)  # 稀疏矩阵全部替换为 1
    array = sparse_array.toarray()  # 返回密集矩阵

    return array


def DivideNet(MatrixAdajacency, ratioTest):
    """
    划分训练集和测试集
    保证网络训练集连通
    """
    # 测试集的连边数目
    num_testlinks = int(ratioTest * np.count_nonzero(MatrixAdajacency) / 2)
    # 将邻接矩阵中所有的边找出来 存储到 linklist 中
    linklist = [list(i) for i in np.argwhere(np.triu(MatrixAdajacency))]
    # 为每条边都设置标志位 ， 判断是否能够删除
    MatrixAdajacency_test = np.zeros(shape=(np.shape(MatrixAdajacency)[0], np.shape(MatrixAdajacency)[1]))

    while np.count_nonzero(MatrixAdajacency_test) < num_testlinks:
        #  ## 随机选择一条边
        link_index = int(np.random.rand(1) * len(linklist))

        uid1 = linklist[link_index][0]  # 边两端的节点 1
        uid2 = linklist[link_index][1]

        #  ## 判断所选边两端节点 uid1 和 uid2 是否可达， 若可达则放入测试集， 否则重新选边
        # 将这条边从网络中挖去，
        MatrixAdajacency[uid1, uid2] = 0
        MatrixAdajacency[uid2, uid1] = 0

        tempvector = MatrixAdajacency[uid1]  # 取出 uid1  一步可达的点 构成一维向量
        sign = 0  # 标记此边是否可以被移除，  sign = 0 表示不可， sign = 1 表示可以
        uid1TOuid2 = np.dot(tempvector, MatrixAdajacency) + tempvector  # 表示 uid1 2步内可达的点

        if uid1TOuid2[0, uid2] > 0:
            sign = 1  # 两步即可到达
        else:
            count = 1
            while len((spones(uid1TOuid2) - tempvector).nonzero()[0]) != 0:
                # 直到可达的点到达稳定状态， 仍然不能到达 uid2， 此边就不能删除
                tempvector = spones(uid1TOuid2)
                uid1TOuid2 = np.dot(tempvector, MatrixAdajacency) + tempvector  # 表示 K 步 内可达的点
                count += 1
                if uid1TOuid2[0, uid2] > 0:
                    sign = 1  # 某步内可以到达
                    break

                if count >= MatrixAdajacency.shape[0]:
                    print("不可达" + str([uid1, uid2]))
                    sign = 0

        #  ## 如果边可以删除， 将其放入测试集中， 并从 link 集中删除
        if sign == 1:
            linklist.pop(link_index)
            MatrixAdajacency_test[uid1, uid2] = 1

        #  ## 如果不可以删除， 恢复原始矩阵， 也从 link 集中删除
        else:
            linklist.pop(link_index)
            MatrixAdajacency[uid1, uid2] = 1
            MatrixAdajacency[uid2, uid1] = 1

    MatrixAdajacency_Train = MatrixAdajacency
    MatrixAdajacency_Test = MatrixAdajacency_test + MatrixAdajacency_test.T

    return MatrixAdajacency_Train, MatrixAdajacency_Test
