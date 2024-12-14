import torch
from utils import perturb_graph
import argparse
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import utils
from attack_models.grl_models import AttackModel
import pickle
from torch.multiprocessing import Process, Array, set_start_method, Value, Pool
import os 
import random
import time
import pandas as pd
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class Perturbations:
    def __init__(self):
        self.node_pairs = []
        self.delta_node_features = []

    def get_node_pairs (self):
        return torch.tensor(self.node_pairs).long()
    
    def add_node_pair(self, node1, node2):
        self.node_pairs.append([node1, node2])

def get_attack_model (saved_model, device):#, embs=False):
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
    try:
        model.load_state_dict(torch.load(saved_model+ '.model', map_location=map_location))
    except:
        model.model.load_state_dict(torch.load(saved_model+ '.model', map_location=map_location))
    model.eval()
    return model


def get_actual_out (inp, node_labels, orig_graph, down_task):
    if (down_task == "node_classification"):
        return (node_labels[inp])
    elif (down_task == "link_prediction"):
        n1, n2 = inp
        n1_s, n1_e = torch.searchsorted(orig_graph[0], torch.stack((n1, n1+1)))
        if (n1_s == n1_e):
            return 0
        else:
            np_s, np_e =  torch.tensor([n1_s, n1_s]) + torch.searchsorted(orig_graph[1, n1_s:n1_e], torch.stack((n2, n2+1)))
            if (np_s == np_e):
                return 0
            else:
                return 1
    elif (down_task == "link_pair"):
        return (node_labels[inp[0]] == node_labels[inp[1]]).long().item()
    else:
        return (node_labels[inp])

def perturb_and_acc (acc_arr, i_off, targets, perb_lists, orig_graph, node_features, base_model_name, node_labels, down_task, device, counter, n_total, start_time):
    base_model = load_model (base_model_name, device) 
    for j in range(len(targets)):
        perb_graph = perturb_graph(orig_graph.detach().cpu().clone(), perb_lists[j]).to(device)
        z = base_model(node_features, perb_graph)
        pred_y = base_model.predict(z[targets[j]], classify=True)
        label_y = get_actual_out(targets[j], node_labels, orig_graph.detach().cpu(), down_task)
        # pred_y = base_model.predict(z[idx_test.T], classify=False)
        # pred_y[i_off+j] = roc_auc_score(label_y, torch.sigmoid(pred_y).flatten().data.cpu().numpy())
        acc_arr[i_off+j] = (pred_y == label_y).type(torch.uint8).item()
        # print (i_off+j, score_arr[j], acc_arr[j])
        with counter.get_lock():
            counter.value += 1
        print (counter.value, "/", n_total, "|", time.time() - start_time, end="\r", file=sys.stderr)
    return True


def init_setup(args):
    # taking the first graph since node-level attack only right now.
    root = os.getcwd()
    if (args.dataset in ["cora", "citeseer", "pubmed"]):
        data_root = root + "/data/Planetoid/"
        dataset = Planetoid(root=data_root, name=args.dataset.capitalize(), transform=NormalizeFeatures())[0]
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
    parser.add_argument("-method", type=str, default="gfa", help="gfa/rls2v")
    parser.add_argument("-base_model", type=str, help="gcn, gat, sage")
    parser.add_argument("-dataset", type=str, help="cora, citeseer, pubmed")
    parser.add_argument("-down_task", type=str, default="node_classification")
    parser.add_argument("-budget", type=int, default=20)
    parser.add_argument("-nprocs", type=int, default=10)
    parser.add_argument("-seed", type=int, default=123)
    parser.add_argument("-device", type=str, default='cpu')
    parser.add_argument("-lcc", action='store_true')
    parser.add_argument("-sampled", action='store_true')
    parser.add_argument("-saved_name", type=str, default=None)
    parser.add_argument("-sols_type", type=str, default="csv")

    args, _ = parser.parse_known_args()
    print (args)

    data, idx_train, idx_val, idx_test = init_setup(args)

    root = os.getcwd()
    data_root = root + "/data/Planetoid/" if (args.dataset in ["cora", "citeseer", "pubmed"]) else root + "/data/"

    if ((args.dataset == "pubmed") or (args.sampled)):
        data, idx_test = utils.sample_small_test(data, idx_test, frac=0.1)
        idx_test2 = torch.load(data_root + args.dataset.capitalize() + "/test_smpld_" + args.down_task + "_" + str(args.seed) + ".pt")
        print (torch.all(idx_test == idx_test2))
    
    print (idx_test.shape)
    data = data.to(args.device)
    base_model = load_model(args.base_model, args.device)
    z = base_model (data.x, data.edge_index)
    pred_y = base_model.predict(z[idx_test.T], classify=True)
    acc_test = (pred_y == data.test_y)
    target_mask = (acc_test > 0).detach().cpu()
    test_ids = idx_test[target_mask]
    num_wrong = torch.sum (acc_test == 0)
    print(len(test_ids) / float(len(idx_test)))

    if (args.method == "grand"):
        sols_file = f'test_results/{args.down_task}/{args.dataset}/sols_more_lcc_{args.budget}.csv'
        if (args.saved_name):
            sols_file = f'test_results/{args.down_task}/{args.dataset}/{args.saved_name}.csv'
    else:
        if (args.lcc):
            sols_file = f'baseline_results/{args.method}/{args.down_task}/{args.dataset}/sols_lcc_{args.budget}.csv'
        else:
            sols_file = f'baseline_results/{args.method}/{args.down_task}/{args.dataset}/sols_{args.budget}.csv'
        if (args.saved_name):
            sols_file = f'baseline_results/{args.method}/{args.down_task}/{args.dataset}/{args.saved_name}.csv'
    
    sols_df = pd.read_csv(sols_file)
    if (("PGNN" in args.base_model) and (args.down_task != "node_classification")):
        sampled_idx = torch.randint(len(idx_test), size=(int(0.1*len(idx_test)),))
        data.test_mask = data.test_mask.T[sampled_idx].T
        data.test_y = data.test_y.T[sampled_idx].T
        idx_test = idx_test[sampled_idx]
        print (len(idx_test))
        sols_df = sols_df.loc[sols_df["tin"].isin(sampled_idx.numpy())]

    tindx = True
    sols_df = sols_df.set_index("tin")
    # idx_test_mask = torch.zeros((idx_test.shape[0],), dtype=bool)
    # idx_test_mask[torch.tensor(list(sols_df.index))] = True
    # idx_test = idx_test[idx_test_mask]
    # data.test_y = data.test_y[idx_test_mask]
    for j in range(target_mask.shape[0]):
        if ((target_mask[j]) and (j not in sols_df.index)):
            target_mask[j] = False
    sols_df = sols_df.loc[np.where(target_mask)[0]].reset_index(drop=True)
    
    pred_y = base_model.predict(z[idx_test.T], classify=True)
    acc_test = (pred_y == data.test_y)
    target_mask = acc_test > 0
    test_ids = idx_test[target_mask]
    num_wrong = torch.sum (acc_test == 0)
    print(len(test_ids) / float(len(idx_test)))
    # test_ids = idx_test[target_mask]
    # num_wrong = torch.sum (acc_test == 0)
    # print(len(test_ids) / float(len(idx_test)))

    targets = test_ids
    perb_list = []
    for i in range(len(targets)):
        perb_list.append(Perturbations())

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    for b in range(1, args.budget+1):
        n_total = targets.shape[0]
        accs = Array('i', n_total)
        counter = Value('i', 0)
        proc_ntargets = int (n_total/args.nprocs)
        procs = []
        results = []
        start_time = time.time()
        not_found = []
        for i in range(args.nprocs):
            i_off = proc_ntargets*i
            inds = [j for j in range(proc_ntargets*i, proc_ntargets*(i+1))]
            for j in inds:
                t, v = sols_df.loc[j]["t"], sols_df.loc[j]["u_" + str(b)]
                # only undirected here
                perb_list[j].add_node_pair(v, t)
                perb_list[j].add_node_pair(t, v)
            iperb_lists = [perb_list[j].get_node_pairs() for j in inds]
            iproc = Process(target=perturb_and_acc, 
                            args=(accs, i_off, targets[inds], iperb_lists, data.edge_index, data.x, 
                                args.base_model, data.y, args.down_task, args.device, counter, n_total, start_time))
            iproc.Daemon = True
            iproc.start()
            procs.append(iproc)

        for iproc in procs:
            iproc.join()

        perb_accs = np.frombuffer(accs.get_obj(), dtype="int32")
        acc = np.sum(perb_accs) / (len(test_ids) + num_wrong)
        print('%d | Average test: %d, targets %d, acc %.5f' % (b, len(not_found), len(targets), acc))
        mask = perb_accs == 1
        mask[not_found] = False
        targets = targets[mask]
        perb_list = [perb_list[j] for j in range (len(perb_list)) if (mask[j])]
        sols_df = sols_df.loc[np.where(mask)[0]].reset_index(drop=True)
