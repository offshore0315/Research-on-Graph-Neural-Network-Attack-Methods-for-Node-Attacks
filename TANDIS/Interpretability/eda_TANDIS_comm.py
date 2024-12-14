from networkx.algorithms.assortativity.pairs import node_attribute_xy
from networkx.algorithms.cuts import conductance, edge_expansion, normalized_cut_size, volume
from torch_geometric.datasets import Planetoid
from sklearn.metrics import jaccard_score
import networkx as nx
import os
import itertools
import csv
import numpy as np
import pandas as pd
import pickle
import torch
import math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.simplefilter('error')

def get_input_graph(dataset_name):
    if (dataset_name == 'cora'):
        dataset = Planetoid(root='./Data/', name='Cora')
        return dataset
    elif (dataset_name == 'citeseer'):
        dataset = Planetoid(root='./Data/', name='Citeseer')
        return dataset
    elif (dataset_name == 'pubmed'):
        dataset = Planetoid(root='./Data/', name='PubMed')
        return dataset
    else:
        print("Not defined")
        return None

# get largest connected component
def get_largest_component(data):
    from torch_geometric.utils import to_scipy_sparse_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    import numpy as np
    from deeprobust.graph.utils import to_tensor
    adj = csr_matrix(to_scipy_sparse_matrix(data.edge_index))
    lcc_idx = np.where(connected_components(adj)[1] == 0)
    adj = adj[lcc_idx].T[lcc_idx].T
    data.x = data.x[lcc_idx]
    data.y = data.y[lcc_idx]
    data.train_mask, data.val_mask, data.test_mask = data.train_mask[
        lcc_idx], data.val_mask[lcc_idx], data.test_mask[lcc_idx]
    data.edge_index = to_tensor(data.x, adj)[1]._indices()
    return data


# get dataset
dataset = get_input_graph('cora')
print(dataset.num_classes)
data = get_largest_component(dataset[0])

print(data.x.shape)
x_shape = np.array(data.x.shape)
num_nodes = x_shape[0]
print(num_nodes)


attr = {}
for i in range(num_nodes):
    attr[i] = {"attr": data.x[i], "label": data.y[i]}


def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]
    print(len(values['x'][0]))
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    nx.set_node_attributes(G, node_attrs, "attr")

    return G


# convert to network_x
g = to_networkx(data, attr)
deg = list(g.degree(g.nodes()))
node_attributes = nx.get_node_attributes(g, 'attr')

deg_dict = {}
deg_list = []
for tup in deg:
    deg_dict[tup[0]] = tup[1]
    deg_list.append(tup[1])

mean_degree = np.mean(deg_list)
median_degree = np.median(deg_list)

# getting 2-hop nn of every node for reverse-knn
two_hop_nn = {}
for node in g.nodes():
    one_hop = list(nn for nn in g[node])
    two_hop = set()
    for one_hop_nn in one_hop:
        nn_2 = list(nn for nn in g[one_hop_nn])
        for n in nn_2:
            two_hop.add(n)
    one_hop.extend(list(two_hop))
    one_hop = list(set(one_hop))  # remove duplicates
    one_hop.sort()
    two_hop_nn[node] = one_hop

def reverse_knn():
    freq = {}  # key:node, value: frequency in 2 hop nn of all nodes
    for node in g.nodes():
        freq[node] = 0
        for key in two_hop_nn.keys():
            if(node in two_hop_nn[key]):
                freq[node] += 1              
    return freq

#freq of each node in 2 hop neighbourhood
freq = reverse_knn()

def get_ranks(all_node_pairs):
    ranks = {}
    sorted_by_scores = dict(sorted(all_node_pairs.items(), key=lambda item: item[1][1], reverse=True))
    rank=1
    # print(sorted_by_scores.keys())
    for key in sorted_by_scores.keys():
        if(key[1] != -1):
            ranks[key[1]] = rank
            rank+=1
    # print(ranks)
    return ranks

fp = open('RL_neighbour_score.pickle', 'rb')
node_ranks = pickle.load(fp)
fp.close()

targets = list(node_ranks.keys())
target_comm_props = {}
for target in targets:
    node_pair = (target, -1)  # original 50NN of target in emb space
    nn_50 = node_ranks[target][node_pair][0]
    #print(target, nn_50)
    S = nn_50
    edge_expansion_orig = nx.edge_expansion(g, S)
    conductance_orig = nx.conductance(g, S)
    normalized_cut_size_orig = nx.normalized_cut_size(g, S)
    volume_orig = nx.volume(g, S)
    target_comm_props[target] = {
        'edge-expansion': edge_expansion_orig, 'conductance': conductance_orig, 'normalized-cut-size': normalized_cut_size_orig, 'volume': volume_orig}

def community_props(u, v):
    S = node_ranks[u][(u, v)][0]
    # print(S)
    edge_expansion_perb = nx.edge_expansion(g, S)
    conductance_perb = nx.conductance(g, S)
    normalized_cut_size_perb = nx.normalized_cut_size(g, S)
    volume_perb = nx.volume(g, S)
    return edge_expansion_perb, conductance_perb, normalized_cut_size_perb, volume_perb

field_names = ['Target', 'feat-sim', 'degree', 'reverse-knn-rank', 'local-clustering-coef', 'edge-expansion-diff', 'conductance-diff', 'normalized-cut-size-diff', 'volume_diff']
res = open('correlation.csv', 'w')
res2 = open('pvalue.csv', 'w')
writer = csv.DictWriter(res, fieldnames=field_names)
writer.writeheader()
writer1 = csv.DictWriter(res2, fieldnames=field_names)
writer1.writeheader()

for target in node_ranks.keys():
    print(target)
    all_node_pairs = node_ranks[target]
    rank_list = get_ranks(all_node_pairs)
    node_rank_list = []
    degree = {}
    LCC = {}
    Feat_sim = {}
    Freq2hop = {}
    Edge_exp = {}
    Conductance = {}
    Norm_Cut_Size = {}
    Volume = {}
    for node in rank_list.keys():
        node_rank_list.append(rank_list[node])
        #label_diff = 1 if(torch.equal(node_attributes[tind]['label'], node_attributes[node]['label'])) else 0
        feat_sim = jaccard_score(node_attributes[target]['attr'].numpy(), node_attributes[node]['attr'].numpy())
        deg = deg_dict[node]
        lcc = nx.clustering(g, node)
        freq_2hop = freq[node]
        #community properties
        community_properties = community_props(target, node)
        comm_props_target = target_comm_props[target]
        ee_diff = comm_props_target['edge-expansion'] - \
            community_properties[0]
        c_diff = comm_props_target['conductance'] - community_properties[1]
        ncs_diff = comm_props_target['normalized-cut-size'] - \
            community_properties[2]
        v_diff = comm_props_target['volume'] - community_properties[3]

        #add to respective dictionaries
        degree[node] = deg
        LCC[node] = lcc
        Feat_sim[node] = feat_sim
        Freq2hop[node] = freq_2hop
        Edge_exp[node] = ee_diff
        Conductance[node] = c_diff
        Norm_Cut_Size[node] = ncs_diff
        Volume[node] = v_diff

    #reorder dictionaries for ranking
    degree = dict(sorted(degree.items(), key=lambda item: item[1])) #lower degree is better
    LCC = dict(sorted(LCC.items(), key=lambda item: item[1], reverse=True)) #higher lcc is better
    Freq2hop = dict(sorted(Freq2hop.items(), key=lambda item: item[1], reverse=True)) #higher frequency is better
    Feat_sim = dict(sorted(Feat_sim.items(), key=lambda item: item[1])) #lower feat_sim is better - homophily
    Edge_exp = dict(sorted(Edge_exp.items(), key=lambda item: item[1], reverse=True))
    Conductance = dict(sorted(Conductance.items(), key=lambda item: item[1], reverse=True))
    Norm_Cut_Size = dict(sorted(Norm_Cut_Size.items(), key=lambda item: item[1], reverse=True))
    Volume = dict(sorted(Volume.items(), key=lambda item: item[1], reverse=True))

    rank=1
    for node in degree.keys():
        degree[node] = rank
        rank+=1
    
    rank=1
    for node in LCC.keys():
        LCC[node] = rank
        rank+=1
    
    rank=1
    for node in Freq2hop.keys():
        Freq2hop[node] = rank
        rank+=1

    rank=1
    for node in Feat_sim.keys():
        Feat_sim[node] = rank
        rank+=1
    
    rank=1
    for node in Edge_exp.keys():
        Edge_exp[node] = rank
        rank+=1
    
    rank=1
    for node in Conductance.keys():
        Conductance[node] = rank
        rank+=1

    rank=1
    for node in Norm_Cut_Size.keys():
        Norm_Cut_Size[node] = rank
        rank+=1

    rank=1
    for node in Volume.keys():
        Volume[node] = rank
        rank+=1

    degree_list = []
    LCC_list = []
    Freq2hop_list = []
    Feat_sim_list = []
    Edge_exp_list = []
    Conductance_list = []
    Norm_Cut_Size_list = []
    Volume_list = []

    for node in rank_list.keys():
        degree_list.append(degree[node])
        LCC_list.append(LCC[node])
        Freq2hop_list.append(Freq2hop[node])
        Feat_sim_list.append(Feat_sim[node])
        Edge_exp_list.append(Edge_exp[node])
        Conductance_list.append(Conductance[node])
        Norm_Cut_Size_list.append(Norm_Cut_Size[node])
        Volume_list.append(Volume[node])

    corr_prop = {}
    p_val = {}
    corr_prop['Target'] = target
    p_val['Target'] = target
    try:
        val = spearmanr(node_rank_list, Feat_sim_list)
        corr_prop['feat-sim'] = val[0]
        p_val['feat-sim'] = val[1]
    except: 
        corr_prop['feat-sim'] = 0

    val = spearmanr(node_rank_list, degree_list)
    corr_prop['degree'] = val[0]
    p_val['degree'] = val[1]

    val = spearmanr(node_rank_list, Freq2hop_list)
    corr_prop['reverse-knn-rank'] = val[0]
    p_val['reverse-knn-rank'] = val[1]

    try:
        val = spearmanr(node_rank_list, LCC_list)
        corr_prop['local-clustering-coef'] = val[0]
        p_val['local-clustering-coef'] = val[1]
    except:
        corr_prop['local-clustering-coef'] = 0

    val = spearmanr(node_rank_list, Edge_exp_list)
    corr_prop['edge-expansion-diff'] = val[0]
    p_val['edge-expansion-diff'] = val[1]

    val = spearmanr(node_rank_list, Conductance_list)
    corr_prop['conductance-diff'] = val[0]
    p_val['conductance-diff'] = val[1]

    val = spearmanr(node_rank_list, Norm_Cut_Size_list)
    corr_prop['normalized-cut-size-diff'] = val[0]
    p_val['normalized-cut-size-diff'] = val[1]

    val = spearmanr(node_rank_list, Volume_list)
    corr_prop['volume_diff'] = val[0]
    p_val['volume_diff'] = val[1]

    writer.writerow(corr_prop)
    writer1.writerow(p_val)
#     #break
res.close()
res2.close()



        


