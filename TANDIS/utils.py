import torch_geometric.utils as tg_utils
import numpy as np
import pickle as pkl
import networkx as nx
import torch
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os

from cmd_args import cmd_args

def random_split (data, num_classes, num_train_per_class, num_val, num_test):
    data.train_mask.fill_(False)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True

    remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    data.val_mask.fill_(False)
    data.val_mask[remaining[:num_val]] = True

    data.test_mask.fill_(False)
    data.test_mask[remaining[num_val:num_val + num_test]] = True

    return data

def perturb_graph (graph, node_pairs):
    for i in range(node_pairs.shape[0]):
        n1, n2 = node_pairs[i, 0], node_pairs[i, 1]
        edge_add = node_pairs[i].reshape((2, 1))
        n1_s, n1_e = torch.searchsorted(graph[0], torch.stack((n1, n1+1)))
        # n1_s, n1_e = torch.searchsorted(graph[0], torch.stack((n1, n1+1)).contiguous())

        if (n1_s == n1_e):
            # node pair does not have an edge => edge addition
            graph = torch.cat ((graph[:, :n1_s], edge_add, graph[:, n1_s:]), dim=1)
        else:
            # relative start and end points
            np_s, np_e =  torch.tensor([n1_s, n1_s]) + torch.searchsorted(graph[1, n1_s:n1_e], torch.stack((n2, n2+1)))
            if (np_s == np_e):
                # node pair does not have an edge => edge addition
                graph = torch.cat ((graph[:, :np_s], edge_add, graph[:, np_s:]), dim=1)
            else:
                # edge deletion
                assert (np_e == (np_s + 1))
                graph = torch.cat((graph[:, :np_s], graph[:, np_e:]), dim=1)
    return graph

def load_txt_data(data_folder, dataset_str):
    idx_train = list(np.loadtxt(data_folder + '/train_idx.txt', dtype=int))
    idx_val = list(np.loadtxt(data_folder + '/val_idx.txt', dtype=int))
    idx_test = list(np.loadtxt(data_folder + '/test_idx.txt', dtype=int))
    return idx_train, idx_val, idx_test

def sample_pos_neg (labels, sample_size, edge_index, nnodes, paired=False):
    while (True):
        pos_inds = torch.randint(edge_index.shape[1], (sample_size,))
        pos_edges = edge_index[:, pos_inds]
        neg_edges = tg_utils.negative_sampling(edge_index, num_neg_samples=sample_size)
        mask = torch.cat ((pos_edges, neg_edges), dim=1)
        if (paired):
            y = torch.cat(((labels[pos_edges[0,:]] == labels[pos_edges[1,:]]).long(), (labels[neg_edges[0,:]] == labels[neg_edges[1,:]]).long()))
        else:
            y = torch.cat((torch.ones(pos_edges.shape[1], dtype=torch.int64), torch.zeros(neg_edges.shape[1], dtype=torch.int64)))
        if (torch.all(mask <= nnodes)):
            break
    return pos_inds, mask, y
    
def edges_split (data, paired=False, train_p=0.1, val_p=0.1, test_p=0.3):
    # split into existent and non-existent edges
    # 
    edge_index = data.edge_index
    nnodes = data.x.shape[0]
    pos_inds, data.train_mask, data.train_y = sample_pos_neg (data.y, int(train_p * edge_index.shape[1]), edge_index, nnodes, paired=paired)
    #
    all_inds = np.arange(edge_index.shape[1])
    all_inds = np.delete(all_inds, pos_inds.detach().numpy())
    edge_index = edge_index[:, all_inds]
    pos_inds, data.val_mask, data.val_y = sample_pos_neg (data.y, int(val_p * edge_index.shape[1]), edge_index, nnodes, paired=paired)
    #
    all_inds = np.arange(edge_index.shape[1])
    all_inds = np.delete(all_inds, pos_inds.detach().numpy())
    edge_index = edge_index[:, all_inds]
    _, data.test_mask, data.test_y = sample_pos_neg (data.y, int(test_p * edge_index.shape[1]), edge_index, nnodes, paired=paired)
    # For BCE loss 
    data.train_y = data.train_y.float()
    data.val_y = data.val_y.float()
    data.test_y = data.test_y.float()
    return data

def split_frac (data, data_root, train_p=0.1, val_p=0.1, test_p=0.8):
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    from cmd_args import cmd_args
    nnodes, nclasses = data.x.shape[0], data.y.unique().shape[0]
    num_val, num_test=int(val_p*nnodes), int(test_p*nnodes)
    num_train_per_class=int(train_p*nnodes/nclasses)
    data = random_split(data, nclasses, num_train_per_class, num_val, num_test)
    # data = Dataset(root=data_root, name=cmd_args.dataset.capitalize(), split='random', num_val=int(val_p*nnodes), 
    #                 num_test=int(test_p*nnodes), num_train_per_class=int(train_p*nnodes/nclasses), transform=NormalizeFeatures())[0]
    data.train_mask = torch.where(data.train_mask)[0]
    data.val_mask = torch.where(data.val_mask)[0]
    data.test_mask = torch.where(data.test_mask)[0]
    data.train_y = data.y[data.train_mask]
    data.val_y = data.y[data.val_mask]
    data.test_y = data.y[data.test_mask]
    return data

def mask_fix (data):
    data.train_mask = torch.where(data.train_mask)[0]
    data.val_mask = torch.where(data.val_mask)[0]
    data.test_mask = torch.where(data.test_mask)[0]
    data.train_y = data.y[data.train_mask]
    data.val_y = data.y[data.val_mask]
    data.test_y = data.y[data.test_mask]
    return data

def preprocess (data, data_root, down_task, train_p=0.1, val_p=0.1, test_p=0.8):
    if (down_task == "link_prediction"):
        test_p=0.3
        data = edges_split (data, train_p=train_p, val_p=val_p, test_p=test_p)
    elif (down_task == "link_pair"):
        test_p=0.3
        data = edges_split (data, paired=True, train_p=train_p, val_p=val_p, test_p=test_p)
    elif (down_task == "node_classification"):
        data = split_frac(data, data_root, train_p=train_p, val_p=val_p, test_p=test_p) #
        # data = mask_fix(data) #
    elif (down_task == "unsupervised"):
        train_p, val_p, test_p = 0.4, 0.0, 0.6
        data = split_frac(data, data_root, train_p=train_p, val_p=val_p, test_p=test_p) #
    return data

def sample_small_test (data, idx_test, frac=0.1):
    # np.random.seed(87)
    # torch.manual_seed(87)
    sampled_idx = torch.randint(len(idx_test), size=(int(frac*len(idx_test)),))
    data.test_mask = data.test_mask.T[sampled_idx].T
    data.test_y = data.test_y.T[sampled_idx].T
    idx_test = idx_test[sampled_idx]
    return data, idx_test

def load_other_datasets (data_root):
    from deeprobust.graph.data import Dataset
    from deeprobust.graph.utils import to_tensor
    data = Dataset(root=data_root, name=cmd_args.dataset, setting="gcn")
    adj, features, labels = to_tensor(data.adj, data.features, data.labels)
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    data.edge_index = adj._indices().to(cmd_args.device)
    data.x = features.to_dense().to(cmd_args.device)
    data.y = labels
    data.adj, data.features, data.labels = None, None, None
    data.train_mask = torch.zeros(data.x.shape[0], dtype=bool)
    data.val_mask = torch.zeros(data.x.shape[0], dtype=bool)
    data.test_mask = torch.zeros(data.x.shape[0], dtype=bool)
    data.train_mask[idx_train], data.val_mask[idx_val], data.test_mask[idx_val] = True, True, True
    return data


def get_largest_component (data):
    from torch_geometric.utils import to_scipy_sparse_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    import numpy as np
    from deeprobust.graph.utils import to_tensor
    adj = csr_matrix(to_scipy_sparse_matrix(data.edge_index))
    ncc, cc_map = connected_components (adj)
    count_cc = np.zeros(shape=ncc, dtype=np.int64)
    for i in range(ncc):
        count_cc[i] = np.sum(cc_map == i)
    lcc_idx = np.where(cc_map == np.argmax(count_cc))[0]
    adj = adj[lcc_idx].T[lcc_idx].T
    data.x = data.x[lcc_idx]
    data.y = data.y[lcc_idx]
    data.train_mask, data.val_mask, data.test_mask = data.train_mask[lcc_idx], data.val_mask[lcc_idx], data.test_mask[lcc_idx]
    data.edge_index = to_tensor(data.x, adj)[1]._indices()
    return data