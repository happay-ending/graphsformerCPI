import json
import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset

from data_process import target_to_cmap
from utils import pad_array


def normalize(mx):
    rowsum = np.array(mx.sum(1) + 1e-10)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)  # 此处得到一个对角矩阵
    mx = r_mat_inv.dot(mx)  # 注意.dot为矩阵乘法,不是对应元素相乘
    return mx


def normalize_adj(mx):
    '''D^(-1/2)AD^(-1/2)'''
    np.fill_diagonal(mx, 1)
    d = np.array(mx.sum(1))
    r_inv_sqrt = np.power(d, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    mid = np.dot(r_mat_inv_sqrt, mx)
    return np.dot(mid, r_mat_inv_sqrt)


class Regression_DataSet(Dataset):
    """自定义数据集"""

    def __init__(self, root, sample_ids, cpd_cut_len, prt_cut_len):
        self.sample_ids = sample_ids
        self.prts = json.load(open(os.path.join(root, 'proteins(revised).txt')),
                              object_pairs_hook=OrderedDict)  # nem*[pid,sequene]
        self.cpds = json.load(open(os.path.join(root, 'ligands_can.txt')),
                              object_pairs_hook=OrderedDict)  # nem*[cid,smiles]
        self.cpd_graphs_dir = os.path.join(root, 'cpd_graph')
        self.prt_cmap_dir = os.path.join(root, 'cmap')
        self.cpd_cut_len = cpd_cut_len
        self.prt_cut_len = prt_cut_len  # 获取蛋白质的最大长度，准备截断

        affinity = pickle.load(open(os.path.join(root, 'Y'), 'rb'), encoding='latin1')
        if 'davis' in root:
            affinity = [-np.log10(y / 1e9) for y in affinity]
        self.affinity = np.asarray(affinity)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, item):
        entry = self.sample_ids[item]

        rows, cols = np.where(np.isnan(self.affinity) == False)
        row, col = rows[entry], cols[entry]

        cpd_npzfile = np.load(os.path.join(self.cpd_graphs_dir, '{0}.npz'.format(list(self.cpds.keys())[row])))
        cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix = cpd_npzfile['atom_features'], cpd_npzfile['adj_matrix'], \
                                                             cpd_npzfile[
                                                                 'dist_matrix']
        cpd_size = cpd_atom_features.shape[0]

        prt_graph = target_to_cmap(list(self.prts.keys())[col], self.prts[list(self.prts.keys())[col]],
                                   self.prt_cmap_dir)  # target_size, target_feature, contact_map, dist_matrix

        cid = list(self.cpds.keys())[row]
        uniprot_id = list(self.prts.keys())[col]
        affinity_label = self.affinity[row, col]

        prt_size, prt_aa_features, prt_contact_map, prt_dist_matrix = prt_graph

        # std_sca = StandardScaler()
        # std_sca = MinMaxScaler()
        # cpd_atom_features = std_sca.fit_transform(cpd_atom_features.T).T
        # cpd_adj_matrix = std_sca.fit_transform(cpd_adj_matrix.T).T
        # cpd_dist_matrix = std_sca.fit_transform(cpd_dist_matrix.T).T

        # 这里可以测试按行标准化
        # cpd_adj_matrix = std_sca.fit_transform(cpd_adj_matrix.reshape(-1,1)).reshape(cpd_adj_matrix.shape)
        # cpd_dist_matrix = std_sca.fit_transform(cpd_dist_matrix.reshape(-1, 1)).reshape(cpd_dist_matrix.shape)

        # prt_aa_features = std_sca.fit_transform(prt_aa_features.T).T
        # prt_contact_map = std_sca.fit_transform(prt_contact_map.T).T
        # prt_dist_matrix = std_sca.fit_transform(prt_dist_matrix.T).T

        # 这里可以测试按行标准化
        # prt_contact_map = std_sca.fit_transform(prt_contact_map.reshape(-1, 1)).reshape(prt_contact_map.shape)
        # prt_dist_matrix = std_sca.fit_transform(prt_dist_matrix.reshape(-1, 1)).reshape(prt_dist_matrix.shape)

        # if prt_aa_features.shape[0]>self.prt_cut_len:
        #     prt_aa_features = prt_aa_features[:self.prt_cut_len, :]
        #     prt_contact_map = prt_contact_map[:self.prt_cut_len, :self.prt_cut_len]
        #     prt_dist_matrix = prt_dist_matrix[:self.prt_cut_len, :self.prt_cut_len]
        if prt_size > self.prt_cut_len:
            prt_aa_features = prt_aa_features[:self.prt_cut_len, :]
            prt_contact_map = prt_contact_map[:self.prt_cut_len, :self.prt_cut_len]
            prt_dist_matrix = prt_dist_matrix[:self.prt_cut_len, :self.prt_cut_len]
        elif prt_size < self.prt_cut_len:
            prt_aa_features = pad_array(prt_aa_features, (self.prt_cut_len, prt_aa_features.shape[1]))
            prt_contact_map = pad_array(prt_contact_map, (self.prt_cut_len, self.prt_cut_len))
            prt_dist_matrix = pad_array(prt_dist_matrix, (self.prt_cut_len, self.prt_cut_len))

        if cpd_size > self.cpd_cut_len:
            cpd_atom_features = cpd_atom_features[:self.cpd_cut_len, :]
            cpd_adj_matrix = cpd_adj_matrix[:self.cpd_cut_len, :self.cpd_cut_len]
            cpd_dist_matrix = cpd_dist_matrix[:self.cpd_cut_len, :self.cpd_cut_len]
        elif cpd_size < self.cpd_cut_len:
            cpd_atom_features = pad_array(cpd_atom_features, (self.cpd_cut_len, cpd_atom_features.shape[1]))
            cpd_adj_matrix = pad_array(cpd_adj_matrix, (self.cpd_cut_len, self.cpd_cut_len))
            cpd_dist_matrix = pad_array(cpd_dist_matrix, (self.cpd_cut_len, self.cpd_cut_len))

        '''对特征矩阵进行归一化'''
        cpd_atom_features = normalize(cpd_atom_features)
        prt_aa_features = normalize(prt_aa_features)
        '''对邻接矩阵进行正则化 D^(-1/2)AD^(-1/2)'''
        cpd_adj_matrix = normalize_adj(cpd_adj_matrix)
        prt_contact_map = normalize_adj(prt_contact_map)

        return [cid,uniprot_id,cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix, prt_aa_features, prt_contact_map, prt_dist_matrix,
                affinity_label]

    @staticmethod
    def collate_fn(batch):
        cid,uniprot_id,cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix, prt_aa_features, prt_contact_map, prt_dist_matrix, labels = map(
            list, zip(*batch))

        return [np.array(cpd_atom_features), np.array(cpd_adj_matrix), np.array(cpd_dist_matrix),
                np.array(prt_aa_features), np.array(prt_contact_map), np.array(prt_dist_matrix),np.array(cid),np.array(uniprot_id)], torch.tensor(labels,
                                                                                                               dtype=torch.float32).reshape(
            -1, 1)


# class Regression_DataSet(Dataset):
#     """自定义数据集"""
#
#     def __init__(self, sample_ids, cpd_cut_len, cpd_graphs, prt_cut_len, prt_graphs, prt_loc_2_id, affinity):
#         self.prt_loc_2_id = prt_loc_2_id
#         self.sample_ids = sample_ids
#         self.affinity = affinity
#         # self.prts = json.load(open(os.path.join(self.root_path, 'proteins.txt')), object_pairs_hook=OrderedDict) # nem*[pid,sequene]
#         # self.cpds = json.load(open(os.path.join(self.root_path, 'ligands_can.txt')), object_pairs_hook=OrderedDict) # nem*[cid,smiles]
#         self.cpd_graphs = cpd_graphs
#         self.prt_graphs = prt_graphs
#         self.cpd_cut_len = cpd_cut_len
#         self.prt_cut_len = prt_cut_len  # 获取蛋白质的最大长度，准备截断
#
#         # self.cpd_info = self.get_cpd_info(self.cpds, cpd_threshold)
#
#     def __len__(self):
#         return len(self.sample_ids)
#
#     def __getitem__(self, item):
#         entry = self.sample_ids[item]
#
#         rows, cols = np.where(np.isnan(self.affinity) == False)
#         row, col = rows[entry], cols[entry]
#
#         # smile = self.cpds[list(self.cpds.keys())[row]]
#         # smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=True)
#
#         # print("__getitem__中打印设定的化合物最大原子数：",self.cpd_cut_len)
#         # print("__getitem__中打印设定的蛋白质最大长度：",self.prt_cut_len)
#
#         cpd_graph = self.cpd_graphs[list(self.cpd_graphs.keys())[row]]
#
#         prt_id = self.prt_loc_2_id[col]
#         prt_graph = self.prt_graphs[prt_id]
#         # contact_path = os.path.join(self.root_path, 'cmap')
#         # prt_graph = target_to_cmap(prt_id, sequence, contact_path)  # target_size, target_feature, contact_map, dist_matrix
#
#         affinity_label = self.affinity[row, col]
#
#         cpd_size, cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix = cpd_graph
#         prt_size, prt_aa_features, prt_contact_map, prt_dist_matrix = prt_graph
#
#         # std_sca = StandardScaler()
#         # std_sca = MinMaxScaler()
#         # cpd_atom_features = std_sca.fit_transform(cpd_atom_features.T).T
#         # cpd_adj_matrix = std_sca.fit_transform(cpd_adj_matrix.T).T
#         # cpd_dist_matrix = std_sca.fit_transform(cpd_dist_matrix.T).T
#
#         # 这里可以测试按行标准化
#         # cpd_adj_matrix = std_sca.fit_transform(cpd_adj_matrix.reshape(-1,1)).reshape(cpd_adj_matrix.shape)
#         # cpd_dist_matrix = std_sca.fit_transform(cpd_dist_matrix.reshape(-1, 1)).reshape(cpd_dist_matrix.shape)
#
#         # prt_aa_features = std_sca.fit_transform(prt_aa_features.T).T
#         # prt_contact_map = std_sca.fit_transform(prt_contact_map.T).T
#         # prt_dist_matrix = std_sca.fit_transform(prt_dist_matrix.T).T
#
#         # 这里可以测试按行标准化
#         # prt_contact_map = std_sca.fit_transform(prt_contact_map.reshape(-1, 1)).reshape(prt_contact_map.shape)
#         # prt_dist_matrix = std_sca.fit_transform(prt_dist_matrix.reshape(-1, 1)).reshape(prt_dist_matrix.shape)
#
#         # if prt_aa_features.shape[0]>self.prt_cut_len:
#         #     prt_aa_features = prt_aa_features[:self.prt_cut_len, :]
#         #     prt_contact_map = prt_contact_map[:self.prt_cut_len, :self.prt_cut_len]
#         #     prt_dist_matrix = prt_dist_matrix[:self.prt_cut_len, :self.prt_cut_len]
#         if prt_size > self.prt_cut_len:
#             prt_aa_features = prt_aa_features[:self.prt_cut_len, :]
#             prt_contact_map = prt_contact_map[:self.prt_cut_len, :self.prt_cut_len]
#             prt_dist_matrix = prt_dist_matrix[:self.prt_cut_len, :self.prt_cut_len]
#         elif prt_size < self.prt_cut_len:
#             prt_aa_features = pad_array(prt_aa_features, (self.prt_cut_len, prt_aa_features.shape[1]))
#             prt_contact_map = pad_array(prt_contact_map, (self.prt_cut_len, self.prt_cut_len))
#             prt_dist_matrix = pad_array(prt_dist_matrix, (self.prt_cut_len, self.prt_cut_len))
#
#         if cpd_size > self.cpd_cut_len:
#             cpd_atom_features = cpd_atom_features[:self.cpd_cut_len, :]
#             cpd_adj_matrix = cpd_adj_matrix[:self.cpd_cut_len, :self.cpd_cut_len]
#             cpd_dist_matrix = cpd_dist_matrix[:self.cpd_cut_len, :self.cpd_cut_len]
#         elif cpd_size < self.cpd_cut_len:
#             cpd_atom_features = pad_array(cpd_atom_features, (self.cpd_cut_len, cpd_atom_features.shape[1]))
#             cpd_adj_matrix = pad_array(cpd_adj_matrix, (self.cpd_cut_len, self.cpd_cut_len))
#             cpd_dist_matrix = pad_array(cpd_dist_matrix, (self.cpd_cut_len, self.cpd_cut_len))
#
#         '''对特征矩阵进行归一化'''
#         cpd_atom_features = normalize(cpd_atom_features)
#         prt_aa_features = normalize(prt_aa_features)
#         '''对邻接矩阵进行正则化 D^(-1/2)AD^(-1/2)'''
#         cpd_adj_matrix = normalize_adj(cpd_adj_matrix)
#         prt_contact_map = normalize_adj(prt_contact_map)
#
#         return [cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix, prt_aa_features, prt_contact_map, prt_dist_matrix,
#                 affinity_label]
#
#     @staticmethod
#     def collate_fn(batch):
#         cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix, prt_aa_features, prt_contact_map, prt_dist_matrix, labels = map(
#             list, zip(*batch))
#
#         return [np.array(cpd_atom_features), np.array(cpd_adj_matrix), np.array(cpd_dist_matrix),
#                 np.array(prt_aa_features), np.array(prt_contact_map), np.array(prt_dist_matrix)], torch.tensor(labels,
#                                                                                                                dtype=torch.float32).reshape(
#             -1, 1)

class Classified_DataSet(Dataset):
    """自定义数据集"""

    def __init__(self, root, sample_ids, cpd_cut_len, prt_cut_len):
        self.sample_ids = sample_ids
        self.dataset = pd.read_csv(os.path.join(root, root.split('/')[-1] + '_data(6).csv'), header=0)
        # self.prts = json.load(open(os.path.join(self.root_path, 'proteins.txt')), object_pairs_hook=OrderedDict) # nem*[pid,sequene]
        # self.cpds = json.load(open(os.path.join(self.root_path, 'ligands_can.txt')), object_pairs_hook=OrderedDict) # nem*[cid,smiles]
        self.cpd_graphs_dir = os.path.join(root, 'cpd_graph')
        self.prt_cmap_dir = os.path.join(root, 'cmap')
        self.cpd_cut_len = cpd_cut_len
        self.prt_cut_len = prt_cut_len  # 获取蛋白质的最大长度，准备截断

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, item):
        index = self.sample_ids[item]
        entry = self.dataset.iloc[index]
        npzfile = np.load(os.path.join(self.cpd_graphs_dir, entry['cid'] + '.npz'))
        cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix = npzfile['atom_features'], npzfile['adj_matrix'], npzfile[
            'dist_matrix']
        cpd_size = cpd_atom_features.shape[0]

        prt_graph = target_to_cmap(entry['uniprot_id'], entry['seq'], self.prt_cmap_dir)
        prt_size, prt_aa_features, prt_contact_map, prt_dist_matrix = prt_graph

        class_label = entry['label']

        # std_sca = StandardScaler()
        # std_sca = MinMaxScaler()
        # cpd_atom_features = std_sca.fit_transform(cpd_atom_features.T).T
        # cpd_adj_matrix = std_sca.fit_transform(cpd_adj_matrix.T).T
        # cpd_dist_matrix = std_sca.fit_transform(cpd_dist_matrix.T).T

        # 这里可以测试按行标准化
        # cpd_adj_matrix = std_sca.fit_transform(cpd_adj_matrix.reshape(-1,1)).reshape(cpd_adj_matrix.shape)
        # cpd_dist_matrix = std_sca.fit_transform(cpd_dist_matrix.reshape(-1, 1)).reshape(cpd_dist_matrix.shape)

        # prt_aa_features = std_sca.fit_transform(prt_aa_features.T).T
        # prt_contact_map = std_sca.fit_transform(prt_contact_map.T).T
        # prt_dist_matrix = std_sca.fit_transform(prt_dist_matrix.T).T

        # 这里可以测试按行标准化
        # prt_contact_map = std_sca.fit_transform(prt_contact_map.reshape(-1, 1)).reshape(prt_contact_map.shape)
        # prt_dist_matrix = std_sca.fit_transform(prt_dist_matrix.reshape(-1, 1)).reshape(prt_dist_matrix.shape)

        # if prt_aa_features.shape[0]>self.prt_cut_len:
        #     prt_aa_features = prt_aa_features[:self.prt_cut_len, :]
        #     prt_contact_map = prt_contact_map[:self.prt_cut_len, :self.prt_cut_len]
        #     prt_dist_matrix = prt_dist_matrix[:self.prt_cut_len, :self.prt_cut_len]
        if prt_size > self.prt_cut_len:
            prt_aa_features = prt_aa_features[:self.prt_cut_len, :]
            prt_contact_map = prt_contact_map[:self.prt_cut_len, :self.prt_cut_len]
            prt_dist_matrix = prt_dist_matrix[:self.prt_cut_len, :self.prt_cut_len]
        elif prt_size < self.prt_cut_len:
            prt_aa_features = pad_array(prt_aa_features, (self.prt_cut_len, prt_aa_features.shape[1]))
            prt_contact_map = pad_array(prt_contact_map, (self.prt_cut_len, self.prt_cut_len))
            prt_dist_matrix = pad_array(prt_dist_matrix, (self.prt_cut_len, self.prt_cut_len))

        if cpd_size > self.cpd_cut_len:
            cpd_atom_features = cpd_atom_features[:self.cpd_cut_len, :]
            cpd_adj_matrix = cpd_adj_matrix[:self.cpd_cut_len, :self.cpd_cut_len]
            cpd_dist_matrix = cpd_dist_matrix[:self.cpd_cut_len, :self.cpd_cut_len]
        elif cpd_size < self.cpd_cut_len:
            cpd_atom_features = pad_array(cpd_atom_features, (self.cpd_cut_len, cpd_atom_features.shape[1]))
            cpd_adj_matrix = pad_array(cpd_adj_matrix, (self.cpd_cut_len, self.cpd_cut_len))
            cpd_dist_matrix = pad_array(cpd_dist_matrix, (self.cpd_cut_len, self.cpd_cut_len))

        '''对特征矩阵进行归一化'''
        cpd_atom_features = normalize(cpd_atom_features)
        prt_aa_features = normalize(prt_aa_features)
        '''对邻接矩阵进行正则化 D^(-1/2)AD^(-1/2)'''
        cpd_adj_matrix = normalize_adj(cpd_adj_matrix)
        prt_contact_map = normalize_adj(prt_contact_map)

        return [entry['cid'],entry['uniprot_id'],cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix, prt_aa_features, prt_contact_map, prt_dist_matrix,
                class_label]

    @staticmethod
    def collate_fn(batch):
        cid,uniprot_id,cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix, prt_aa_features, prt_contact_map, prt_dist_matrix, labels = map(
            list, zip(*batch))

        return [np.array(cpd_atom_features), np.array(cpd_adj_matrix), np.array(cpd_dist_matrix),
                np.array(prt_aa_features), np.array(prt_contact_map), np.array(prt_dist_matrix),np.array(cid),np.array(uniprot_id)], torch.tensor(labels)

