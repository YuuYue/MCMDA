import os.path
import random
from sklearn.model_selection import KFold
import numpy as np
import scipy.sparse as sp
from utils import *

'''
构建sc view下的dm、md数据
以及相似度数据d_sim_d、m_sim_m
'''
def sc_data(adj,D,M,parent_dir,ds,ms):
    dm = np.array(adj)
    path_dm = os.path.join(parent_dir, "dm.txt")
    np.savetxt(path_dm, dm, fmt="%d")
    dm = sp.csr_matrix((dm[:, 2], (dm[:, 0], dm[:, 1])), shape=(D + M, D + M)).toarray()
    adj_similiarty = np.vstack((np.hstack((ds, np.zeros((D, M)))), np.hstack((np.zeros((M, D)),ms))))
    adj_dm = dm + dm.transpose()
    path_sim = os.path.join(parent_dir, "adj_similiarty.npy")
    path_dm = os.path.join(parent_dir, "adj_dm.npy")
    np.save(path_sim, adj_similiarty)
    np.save(path_dm, adj_dm)

    return dm

"""
    construct meta path
    dmd
    mdm
"""
def mp_data(parent_dir,dm,D,M):
    md = dm.transpose()
    dmd = np.matmul(dm, md)
    dmd = sp.coo_matrix(dmd)
    # dmd.data = np.ones(dmd.data.shape[0])
    path_dmd = os.path.join(parent_dir, "dmd.npz")
    sp.save_npz(path_dmd, dmd)

    mdm = np.matmul(md, dm)
    mdm = sp.coo_matrix(mdm)
    # mdm.data = np.ones(mdm.data.shape[0])
    path_mdm = os.path.join(parent_dir, "mdm.npz")
    sp.save_npz(path_mdm, mdm)

    return dmd, mdm

"""
    construct pos sample for contrastive learning
    for each node has 10 pos sample
"""
def mp_pos(dmd, mdm,parent_dir,D,M,pos_num):

    dmd = dmd.A.astype("float32")
    dia_d = sp.dia_matrix((np.ones(D+M), 0), shape=(D+M, D+M)).toarray()


    dd = np.ones((D+M, D+M)) - dia_d
    dmd = dmd * dd  # 将dmd对角线数据置为0
    pos_d = np.zeros((D, D))
    k = 0
    for i in range(D):
        pos_d[i, i] = 1
        one = dmd[i].nonzero()[0]  # one是数组：保存的是与节点i之间存在meta path的目标节点id,不含节点自身
        if len(one) > pos_num - 1:
            oo = np.argsort(-dmd[i, one])  # np.argsort返回从小到大排列的索引值
            sele = one[oo[:pos_num - 1]]  # sele为得到的4个目标id
            pos_d[i, sele] = 1
            k += 1
        else:
            pos_d[i, one] = 1
    pos_d = sp.coo_matrix(pos_d)
    path_pos_d = os.path.join(parent_dir, "pos_d.npz")
    sp.save_npz(path_pos_d, pos_d)


    mdm = mdm.A.astype("float32")
    # microbe
    dia_m = sp.dia_matrix((np.ones(M+D), 0), shape=(M+D, M+D)).toarray()
    mm = np.ones((M+D, M+D)) - dia_m
    mdm = mdm * mm
    pos_m = np.zeros((M, M))
    k = 0

    for j in range(M):
        pos_m[j, j] = 1
        i = j + D
        one = mdm[i].nonzero()[0]
        if len(one) > pos_num - 1:
            oo = np.argsort(-mdm[i, one])
            sele = one[oo[:pos_num - 1]]
            pos_m[j, sele-D] = 1
            k += 1
        else:
            pos_m[j, one-D] = 1
    pos_m = sp.coo_matrix(pos_m)
    path_pos_m = os.path.join(parent_dir, "pos_m.npz")
    sp.save_npz(path_pos_m, pos_m)

"""
数据五折交叉分组
"""
def cross_5_folds(adj, nodefeatures,dataset,drugfeat,microbefeat,seed=1):
    folds = np.ones(adj.shape[0])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    kf = kf.split(folds)
    a = []
    b = [-1, -1]
    for i, data in enumerate(kf):
        a.append(data[1])

    for i in range(5):
        for j in range(len(a[i])):
            folds[a[i][j]] = i

    adj = adj.astype(np.int32).tolist()
    num = 0
    drug_id_list = list(range(len(drugfeat)))
    microbe_id_list = list(range(len(microbefeat)))
    pos_num = len(adj)

    """
    生成负样本1：5
    """
    temp_adj = []
    folds1 = []
    while num < pos_num:
        drug_id = random.choice(drug_id_list)
        microbe_id = random.choice(microbe_id_list)
        neg_pos1 = [drug_id, microbe_id, 1]
        neg_pos = [drug_id, microbe_id, 0]
        if (neg_pos1 in adj) or (neg_pos in temp_adj):
            continue
        temp_adj.append(adj[num])
        folds1.append(folds[num % pos_num])
        temp_adj.append(neg_pos)
        folds1.append(folds[num % pos_num])
        num += 1

    folds1 = np.array(folds1)
    temp_adj = np.array(temp_adj)
    path_data = os.path.join(dataset,"data_"+dataset+".npz")
    np.savez(path_data, drugfeat=drugfeat, microbefeat=microbefeat, adj=temp_adj, folds=folds1, nodefeat=nodefeatures)

"""
    将drug、microbe fetures放入一个矩阵中，多于位置补0
"""
def feat_combine(dm,D,M,drugfeat,microbefeat,parent_dir):
    nodefeatures = np.vstack((np.hstack((drugfeat, np.zeros([D,M]))),np.hstack((np.zeros([M,D]), microbefeat))))
    path_nodefeatures = os.path.join(parent_dir,"nodefeatures.npy")
    np.save(path_nodefeatures, nodefeatures)
    return nodefeatures


def processdata_encoder(dataset, train_positive_inter_pos, pos_num ):
    # seed = 1
    # random.seed(seed)
    cv_i = 0
    np.random.seed(1)
    for pos_adj in train_positive_inter_pos:

        pos_adj = np.array(pos_adj)
        adj = np.ones([len(pos_adj),3])
        adj[:,0] = np.copy(pos_adj[:,0])
        adj[:,1] = np.copy(pos_adj[:,1])
        pos_num = pos_num  # pos sample num for contrastive learning
        root_dir = os.path.join("./data", dataset)
        parent_dir = os.path.join(root_dir,'encoder_' + str(cv_i) )
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        path_ds = os.path.join(root_dir, "drugsimilarity.txt")
        ds = np.loadtxt(path_ds)
        path_ms = os.path.join(root_dir, "microbesimilarity.txt")
        ms = np.loadtxt(path_ms)
        D = len(ds)
        M = len(ms)
        # path_adj = os.path.join(root_dir, "adj.txt")
        # adj = np.genfromtxt(path_adj)
        for i in adj:
            # i[0] -= 1
            # i[1] -= 1
            i[1] += D


        adj = adj.astype("int")
        path_drugfeat = os.path.join(root_dir, "drugfeatures.txt")
        path_microbefeat = os.path.join(root_dir, "microbefeatures.txt")
        drugfeat = np.loadtxt(path_drugfeat)
        microbefeat = np.loadtxt(path_microbefeat)
        np.savetxt(os.path.join(parent_dir, "microbefeatures.txt"), microbefeat)
        np.savetxt(os.path.join(parent_dir, "adj.txt"), adj)
        np.savetxt(os.path.join(parent_dir, "drugsimilarity.txt"), ds)
        np.savetxt(os.path.join(parent_dir, "microbesimilarity.txt"), ms)
        np.savetxt(os.path.join(parent_dir, "drugfeatures.txt"), drugfeat)


        dm = sc_data(adj,D,M,parent_dir,ds,ms)
        dmd, mdm = mp_data(parent_dir,dm, D,M)
        mp_pos(dmd, mdm,parent_dir,D,M,pos_num)
        nodefeatures = feat_combine(dm,D,M,drugfeat,microbefeat,parent_dir)
        # cross_5_folds(adj, nodefeatures,dataset,drugfeat,microbefeat,seed)
        cv_i += 1




