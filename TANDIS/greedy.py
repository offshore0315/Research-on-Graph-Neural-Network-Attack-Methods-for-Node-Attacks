# from silh_score_reward import dist_score
from utils import perturb_graph
import time
import sys
from functools import partial
import argparse
import os
import torch
import torch
from utils import perturb_graph
import argparse
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import utils
from attack_models.grl_models import AttackModel
import pickle
from torch.multiprocessing import Process, Array, set_start_method, Value, Pool, Manager
import os 
import random
import time
import pandas as pd
import sys

import heapq
import numpy as np
import torch


class Perturbations:
    def __init__(self):
        self.node_pairs = []
        self.delta_node_features = []

    def get_node_pairs (self):
        return torch.tensor(self.node_pairs).long()
    
    def add_node_pair(self, node1, node2):
        self.node_pairs.append([node1, node2])

def get_attack_model (saved_model, device):#, embs=False):
    from attack_models.grl_models import AttackModel
    import pickle
    with open('%s-args.pkl' % saved_model, 'rb') as f:
        base_args = pickle.load(f)

    base_args['device'] = device

    return AttackModel(base_args).to(device) #if (not (embs)) else AttackEmbModel(base_args)

def load_model(saved_model, device):
    import torch
    model = get_attack_model (saved_model, device)
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    model.load_state_dict(torch.load(saved_model+ '.model', map_location=map_location))
    model.eval()
    return model

def find_rel_dist (x, y, measure="Euclidean"):
    if (measure == "Euclidean"):
        return torch.linalg.norm(x - y).item()
    elif (measure == "cosine-sim"):
        return (torch.dot(x, y)/(torch.linalg.norm(x)*torch.linalg.norm(y))).item()

def all_knn (embeddings, k, measure="Euclidean"):
    knn_arr = np.zeros(shape=(embeddings.shape[0], k))
    for i in range(embeddings.shape[0]):
        knn_arr[i, :] = heapq.nsmallest(k+1, np.arange(embeddings.shape[0]), 
                lambda j: find_rel_dist(embeddings[j, :], embeddings[i, :], measure=measure))[1:]
    return knn_arr

def knn_embs (embs, n, k=100, measure="Euclidean"):
    knn_arr = np.zeros(shape=k)
    knn_arr = heapq.nsmallest(k+1, np.arange(embs.shape[0]), 
                lambda j: find_rel_dist(embs[j, :], embs[n, :], measure=measure))[1:]
    return knn_arr

def get_neighborhood (embs, node, neighborhood):
    if (neighborhood.endswith("-NN")):
        k = int(neighborhood[:neighborhood.find("-NN")])
        return knn_embs(embs, node, k=k)

def dist_score (ti, zp, z, only_do=False, neighborhood="50-NN", dist_metric="lp", p=2):
    N_t = get_neighborhood(z, ti, neighborhood)
    if (dist_metric == "lp"):
        if (only_do):
            return (torch.sum(torch.linalg.norm(zp[0,:] - zp[1:,:], ord=p, dim=1))/len(N_t))
        else:
            Np_t = get_neighborhood(zp, ti, neighborhood)
            do_t = torch.sum(torch.linalg.norm(zp[ti,:] - zp[N_t,:], ord=p, dim=1))/len(N_t)
            dp_t = torch.sum(torch.linalg.norm(zp[ti,:] - zp[Np_t,:], ord=p, dim=1))/len(Np_t)
            return (do_t - dp_t).item(), N_t, Np_t

def step_target_node (score_arr, i_off, target_node, perb_lists, last_perbs_list, orig_graph, node_features, emb_model_name, nbrhood, device, counter, n_total, start_time, nn):
    emb_model = load_model (emb_model_name, device) if (emb_model_name is not None) else None
    for j in range(len(perb_lists)):
        last_graph = perturb_graph(orig_graph.detach().clone(), last_perbs_list)
        perb_graph = perturb_graph(orig_graph.detach().clone(), perb_lists[j])
        v = perb_lists[j][0, 0].item() if (perb_lists[j][0, 1].item() == target_node) else perb_lists[j][0, 1].item()
        if (emb_model is not None):
            z = emb_model(node_features, last_graph) #if (cmd_args.reward_state == "marginal") else emb_model(node_features, orig_graph) 
            zp = emb_model(node_features, perb_graph)
            res = dist_score(target_node, zp, z, neighborhood=nbrhood)
            score_arr[i_off + j] = res[0]
            nn[(target_node, -1)] = res[1]  # without perturbation
            nn[(target_node, v)] = res[2]  # has to be single edge (u,v)
        with counter.get_lock():
            counter.value += 1
        print (counter.value, "/", n_total, "|", time.time() - start_time, end="\r", file=sys.stderr)
    return True


def init_setup(args):
    # taking the first graph since node-level attack only right now.
    root = os.getcwd()
    if (args.dataset in ["cora", "citeseer", "pubmed"]):
        data_root = root + "/data/Planetoid/"
        dataset = Planetoid(root=data_root, name=args.dataset.capitalize(), transform=NormalizeFeatures())[0].to(args.device)
    else:
        data_root = root + "/data/"
        dataset = utils.load_other_datasets(data_root)
    if (args.lcc):
        dataset = utils.get_largest_component(dataset)
    idx_file_end = args.down_task + "_" + str(args.seed) + ".pt"
    if (args.lcc):
        dataset = utils.get_largest_component(dataset)
        idx_file_end  = "lcc_" + idx_file_end
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = utils.preprocess(dataset, data_root, args.down_task)
    idx_train = dataset.train_mask.T
    idx_val = dataset.val_mask.T
    idx_test = dataset.test_mask.T
    try:
        idx_train2 = torch.load(data_root + args.dataset.capitalize() + "/train_" + idx_file_end)
        idx_val2 = torch.load(data_root + args.dataset.capitalize() + "/val_" + idx_file_end)
        idx_test2 = torch.load(data_root + args.dataset.capitalize() + "/test_" + idx_file_end)
        print(torch.all(idx_train == idx_train2))
        print(torch.all(idx_val == idx_val2))
        print(torch.all(idx_test == idx_test2))
    except:
        torch.save(idx_train, data_root + args.dataset.capitalize() + "/train_" + idx_file_end)
        torch.save(idx_val, data_root + args.dataset.capitalize() + "/val_" + idx_file_end)
        torch.save(idx_test, data_root + args.dataset.capitalize() + "/test_" + idx_file_end)
    return dataset, idx_train, idx_val, idx_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for evaluating different attacks')
    parser.add_argument("-emb_model_name", type=str, default='attack_models/unsupervised/cora/model-Deep-gin')
    parser.add_argument("-dataset", type=str, help="cora, citeseer, pubmed")
    parser.add_argument("-down_task", type=str, default="node_classification")
    parser.add_argument("-budget", type=int, default=20)
    parser.add_argument("-nprocs", type=int, default=10)
    parser.add_argument("-seed", type=int, default=123)
    parser.add_argument("-device", type=str, default='cpu')
    parser.add_argument("-lcc", action='store_true')
    parser.add_argument("-sampled", action='store_true')
    parser.add_argument("-save_scores", action='store_true')
    neighborhood = '50-NN'
    
    args, _ = parser.parse_known_args()

    data, idx_train, idx_val, idx_test = init_setup(args)

    targets = idx_test[:10]
    if (args.down_task == "node_classification"):
        target_nodes = targets
    else:
        target_nodes = torch.zeros(targets.shape[0], dtype=torch.int64)
        for i in range(targets.shape[0]):
            target_nodes[i] = targets[i, torch.randint(2, (1,))]

    perb_list = []
    for i in range(len(targets)):
        perb_list.append(Perturbations())

    try:
        set_start_method('spawn')
    except RuntimeError:
        exit(1)

    solutions = {}

    for b in range(1, args.budget+1):
        if (args.save_scores):
            scores_dict = {}
            nn_dict = {}

        for j in range(targets.shape[0]):
            target, target_node = targets[j].item(), target_nodes[j].item()
            # only undirected here
            # target_node = target_node.item()
            solutions[target_node] = []
            poss_perbs_mask = torch.ones(data.num_nodes, dtype=bool)
            poss_perbs_mask[target] = False
            poss_perbs = torch.where(poss_perbs_mask)[0].numpy()
            proc_nperbs = int (len(poss_perbs)/args.nprocs)
            score_arr = Array('d', len(poss_perbs))
            manager = Manager()
            nn = manager.dict()
            counter = Value('i', 0)
            n_total = len(poss_perbs)
            procs = []
            start_time = time.time()
            last_perbs_list = perb_list[j].get_node_pairs()
            for i in range(args.nprocs):
                i_off = proc_nperbs*i
                inds = [j for j in range(proc_nperbs*i, proc_nperbs*(i+1))]
                iperbs_lists = [torch.tensor([[target_node, v], [v, target_node]]).long() for v in poss_perbs[inds]]
                iproc = Process(target=step_target_node, 
                                args=(score_arr, i_off, target_node, iperbs_lists, last_perbs_list, 
                                    data.edge_index, data.x, args.emb_model_name, neighborhood, args.device, 
                                    counter, n_total, start_time, nn))
                iproc.Daemon = True
                iproc.start()
                procs.append(iproc)

            for iproc in procs:
                iproc.join()

            scores = np.frombuffer(score_arr.get_obj(), dtype="int32")
            nn = dict(nn)
            # **** CHECK THIS ****
            # print (scores.shape)
            # v_selected = poss_perbs[np.argmax(scores)]
            # perb_list[j].add_node_pair(v_selected, target_node)
            # perb_list[j].add_node_pair(target_node, v_selected)

            # solutions[target_node].append(v_selected)
            
            if (args.save_scores):
                for v, score in zip(poss_perbs, scores):
                    scores_dict[(target_node, v)] = score

                for x in nn.keys():
                    nn_dict[x] = nn[x]
                
        if (args.save_scores):
            import pickle
            pickle.dump(scores_dict, open(f"scores_test_b{b}.pkl", "wb"))
            pickle.dump(nn_dict, open(f"neighborhood_test_b{b}.pkl", "wb"))


    # try:
    #     os.makedirs(f"baseline_results/greedy/{args.down_task}/{args.dataset}/")
    # except:
    #     pass

    # with open (f"baseline_results/greedy/{args.down_task}/{args.dataset}/sols_lcc_{budget}.csv", "w+") as writef:
    #     import csv
    #     writer = csv.DictWriter(writef, fieldnames=["tin", "t"] + ["u_" + str(b) for b in range(1, budget+1)])
    #     writer.writeheader()
    #     for i, t in enumerate(targets_nodes):
    #         soln = {}
    #         soln["tin"], soln["t"] = i, t
    #         for b in range(1, args.budget+1):
    #             soln["u_" + str(b)] = solutions[t][b-1]