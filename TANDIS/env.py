import sys
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

from cmd_args import cmd_args

from silh_score_reward import dist_score
from utils import perturb_graph
import time
import sys
from functools import partial
from torch.multiprocessing import Process, Array, Value, Pool
    
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

    base_args['device'] = cmd_args.device

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

def step_target_node (score_arr, acc_arr, i_off, targets, target_nodes, actions, perb_lists, last_perbs_lists, orig_graph, node_features, emb_model_name, base_model_name, node_labels, down_task, nbrhood, device, counter, n_total, start_time):

    emb_model = load_model (emb_model_name, device) if (emb_model_name is not None) else None
    base_model = load_model (base_model_name, device) if (base_model_name is not None) else None
    for j in range(len(target_nodes)):
        last_graph = perturb_graph(orig_graph.detach().clone(), last_perbs_lists[j])
        #print ("Original:", orig_graph.shape)
        perb_graph = perturb_graph(orig_graph.detach().clone(), perb_lists[j])
        #print ("After perturbation:", perb_graph.shape)
        if (emb_model is not None):
            z = emb_model(node_features, last_graph) #if (cmd_args.reward_state == "marginal") else emb_model(node_features, orig_graph) 
            zp = emb_model(node_features, perb_graph)
            # print("Orig embeds:", z)
            # print("Perturbed:", zp)
            score_arr[i_off+j] = dist_score(target_nodes[j], zp, z, neighborhood=nbrhood)
        # new accuracy
        if (base_model is not None):
            z = base_model(node_features, perb_graph)
            pred_y = base_model.predict(z[targets[j]], classify=True)
            label_y = get_actual_out(targets[j], node_labels, orig_graph, down_task)
            acc_arr[i_off+j] = (pred_y == label_y).type(torch.uint8).item()
        # print (i_off+j, score_arr[j], acc_arr[j])
        with counter.get_lock():
            counter.value += 1
        print (counter.value, "/", n_total, "|", time.time() - start_time, end="\r", file=sys.stderr)
    return True


class GraphEnv(object):
    def __init__(self, down_task, features, node_labels, edge_index, emb_model, base_model=None):
        
        # init multiprocessing
        try:
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError as e:
            raise(e)
        
        self.down_task = down_task
        self.features = features
        self.node_labels = node_labels # labels are actual labels
        self.orig_graph = edge_index
        self.directed = cmd_args.directed
        # emb_model is a torch_geometric nn Module
        self.emb_model = emb_model
        self.base_model = base_model
        
        self.multi_init = False

    # -----------
    def step_target_node_npp (_):
        print("here npp")
        
        emb_model = load_model(self.emb_model_name_npp, self.device_npp) if (self.emb_model_name_npp is not None) else None
        
        base_model = load_model(self.base_model_name_npp, self.device_npp) if (self.base_model_name_npp is not None) else None
        
        for j in range(len(self.target_nodes)):
            last_graph = perturb_graph(self.orig_graph.detach().clone(), self.last_perbs_lists_npp[j])
            
            perb_graph = perturb_graph(self.orig_graph.detach().clone(), self.iperb_lists_npp[j])
            
            if (emb_model is not None):
                z = emb_model(self.features, last_graph) 
                zp = emb_model(self.features, perb_graph)
    
                self.score_arr_npp[self.i_off_npp+j] = dist_score(self.target_nodes[j], zp, z, neighborhood=self.nbrhood_npp)
            
            if (base_model is not None):
                z = base_model(self.features, perb_graph)
                pred_y = base_model.predict(z[self.targets[j]], classify=True)
                label_y = get_actual_out(self.targets[j], self.labels, self.orig_graph, self.down_task_npp)
                self.acc_arr_npp[self.i_off_npp+j] = (pred_y == label_y).type(torch.uint8).item()

            with self.counter_npp.get_lock():
                self.counter_npp.value += 1
            print (self.counter_npp.value, "/", self.n_total_npp, "|", time.time() - self.start_time_npp, end="\r", file=sys.stderr)
        return True
    
    # ------------
        
    def get_actual_out (self, inp):
        if (self.down_task == "node_classification"):
            return (self.node_labels[inp])
        elif (self.down_task == "link_prediction"):
            n1, n2 = inp
            n1_s, n1_e = torch.searchsorted(self.orig_graph[0], torch.stack((n1, n1+1)))
            if (n1_s == n1_e):
                return False
            else:
                np_s, np_e =  torch.tensor([n1_s, n1_s]) + torch.searchsorted(self.orig_graph[1, n1_s:n1_e], torch.stack((n2, n2+1)))
                if (np_s == np_e):
                    return False
                else:
                    return True
        else:
            return (self.node_labels[inp])
                    
    def setup(self, targets):
        # There is a combinatorial nature wrt target nodes as well. 
        # there could have been but we are not considering for simplicity. 
        # We could do some future experiment on a colluding target set but it's not easy. 
        # So for now let's only consider targets. We will otherwise have target_sets
        self.targets = targets
        # targets = [T_1, T_2, T_3, ... ] where T_i is a target set 
        #print (self.targets)
        if (self.down_task == "node_classification"):
            self.target_nodes = self.targets
        elif (self.down_task in ["link_prediction", "link_pair"]):
            # in case of a link prediction => T_i = [u, v] directed
            self.target_nodes = torch.zeros(self.targets.shape[0], dtype=torch.int64)
            for i in range(self.targets.shape[0]):
                self.target_nodes[i] = self.targets[i, torch.randint(2, (1,))]
        else:
            self.target_nodes = self.targets
        self.n_step = 0
        self.rewards = None
        self.perb_list = []
        for i in range(len(self.targets)):
            self.perb_list.append(Perturbations())

    def set_scores (self, i_off, res):
        for j in range(len(res[0])):
            self.silh_scores[i_off+j] = res[0][j]
            self.accs[i_off+j] = res[1][j]
        # print (self.silh_scores.mean(), self.accs.mean())

    def update_targets (self):
        # masking 
        mask = self.perb_accuracies == 1
        self.target_nodes = self.target_nodes[mask]
        self.targets = self.targets[mask]
        self.perb_list = [self.perb_list[j] for j in range (len(self.perb_list)) if (mask[j])]

    def step(self, actions):
        # actions are always the direction and the node - (+/- v)
        self.n_step += 1
        calc_rewards = False
        if (cmd_args.reward_state == "final"):
            calc_rewards = self.isTerminal()
        elif (cmd_args.reward_state == "marginal"):
            calc_rewards = True #(self.n_step % 2 == 0)

        if (cmd_args.save_sols_only):
            for i in tqdm(range(len(self.target_nodes))):
                # check
                v, direction = np.abs(actions[i]), np.sign(actions[i])
                last_perbs = self.perb_list[i].get_node_pairs()
                if (self.directed):
                    if (direction == 1):
                        self.perb_list[i].add_node_pair(self.target_nodes[i], v)
                    elif (direction == -1):
                        self.perb_list[i].add_node_pair(v, self.target_nodes[i])
                else:
                    self.perb_list[i].add_node_pair(v, self.target_nodes[i])
                    self.perb_list[i].add_node_pair(self.target_nodes[i], v)
            return
        """
        ****TO DO**** 
        can this loop be parallelized ---
        multiprocessing
        """
        if cmd_args.nprocs == 1:
            # training
            silh_scores = np.zeros (len(self.target_nodes), dtype=np.float64)
            accs = []
            for i in tqdm(range(len(self.target_nodes))):
                # check
                v, direction = np.abs(actions[i]), np.sign(actions[i])
                last_perbs = self.perb_list[i].get_node_pairs()
                if (self.directed):
                    if (direction == 1):
                        self.perb_list[i].add_node_pair(self.target_nodes[i], v)
                    elif (direction == -1):
                        self.perb_list[i].add_node_pair(v, self.target_nodes[i])
                else:
                    self.perb_list[i].add_node_pair(v, self.target_nodes[i])
                    self.perb_list[i].add_node_pair(self.target_nodes[i], v)
                if (calc_rewards):
                    last_graph = perturb_graph(self.orig_graph.detach().clone(), last_perbs)
                    #print ("Original:", self.orig_graph.shape)
                    perb_graph = perturb_graph(self.orig_graph.detach().clone(), self.perb_list[i].get_node_pairs())
                    #print ("After perturbation:", perb_graph.shape)
                    
                    z = self.emb_model(self.features, last_graph) if (cmd_args.reward_state == "marginal") else self.emb_model(self.features, self.orig_graph) 
                    zp = self.emb_model(self.features, perb_graph)
                    # print("Orig embeds:", z)
                    # print("Perturbed:", zp)
                    # as numpy array
                    # need to change silh such that we can accept torch tensor
                    silh_scores[i] = dist_score(self.target_nodes[i], zp, z, neighborhood=cmd_args.nbrhood)
                    # new accuracy
                    if (self.base_model is not None):
                        z = self.base_model(self.features, perb_graph)
                        pred_y = self.base_model.predict(z[self.targets[i]], classify=True)
                        label_y = self.get_actual_out(self.targets[i])
                        acc_perb = (pred_y == label_y).type(torch.uint8) 
                        accs.append(acc_perb)

            if (cmd_args.reward_type == 'emb_silh'):
                self.rewards = silh_scores

            self.perb_accuracies = accs
            return

        # otherwise multiprocessing

        self.silh_scores_npp = Array('d', len(self.target_nodes))
        self.accs_npp = Array('i', len(self.target_nodes))
        self.counter_npp = Value('i', 0)
        n_total = len(self.target_nodes)
        self.n_total_npp = len(self.target_nodes)
        
        silh_scores = Array('d', len(self.target_nodes))
        accs = Array('i', len(self.target_nodes))
        counter = Value('i', 0)
        
        proc_ntargets = int (len(self.target_nodes)/cmd_args.nprocs)
        procs = []
        results = []
        
        start_time = time.time()
        self.start_time_npp = time.time()
        
#         pool = Pool (cmd_args.nprocs)
        emb_model_name = cmd_args.saved_emb_model if (self.emb_model is not None) else None
        self.emb_model_name_npp = cmd_args.saved_emb_model if (self.emb_model is not None) else None
        
        for i in range(cmd_args.nprocs):
            i_off = proc_ntargets*i
            self.i_off_npp = i_off
            
            inds = [j for j in range(proc_ntargets*i, proc_ntargets*(i+1))]
            i_actions = [actions[j] for j in inds]
            last_perbs_list = []
            self.last_perbs_list_npp = []
            for j in inds:
                v, direction = np.abs(actions[j]), np.sign(actions[j])
                
                last_perbs_list.append(self.perb_list[j].get_node_pairs())
                self.last_perbs_list_npp.append(self.perb_list[j].get_node_pairs())
                
                if (self.directed):
                    if (direction == 1):
                        self.perb_list[j].add_node_pair(self.target_nodes[j], v)
                    elif (direction == -1):
                        self.perb_list[j].add_node_pair(v, self.target_nodes[j])
                else:
                    self.perb_list[j].add_node_pair(v, self.target_nodes[j])
                    self.perb_list[j].add_node_pair(self.target_nodes[j], v)
            iperb_lists = [self.perb_list[j].get_node_pairs() for j in inds]
            self.iperb_lists_npp = [self.perb_list[j].get_node_pairs() for j in inds]
            
            iproc = Process(target=step_target_node, 
                            args=(silh_scores, accs, i_off, self.targets[inds], self.target_nodes[inds], i_actions, iperb_lists, last_perbs_list, self.orig_graph, self.features, emb_model_name, cmd_args.saved_model, self.node_labels, self.down_task, cmd_args.nbrhood, cmd_args.device, counter, n_total, start_time))
            iproc.Daemon = True
            iproc.start()
            procs.append(iproc)
        for iproc in procs:
            iproc.join()
            
#         try:
#             self.nbrhood_npp = cmd_args.nbrhood
#             self.device_npp = cmd_args.device
#             self.down_task_npp = self.down_task
#             self.base_model_name_npp = cmd_args.saved_model
#             pool.map(self.step_target_node_npp, (0,0,0,0))
#         finally:
#             pool.close()
#             pool.join()
            
        
        if (cmd_args.reward_type == 'emb_silh'):
            self.rewards = np.frombuffer(silh_scores.get_obj())

        self.perb_accuracies = np.frombuffer(accs.get_obj(), dtype="int32")
        
    def isTerminal(self):
        if self.n_step == cmd_args.budget:
            return True
        return False
    
    def getStateRef(self):
        return list(zip(self.target_nodes.cpu(), self.perb_list))
                        
    def cloneState(self):
        return list(zip(self.target_nodes[:].cpu(), deepcopy(self.perb_list)))
    
    def printState(self):
        for i in range(len(self.perb_list)):
            perb_graph = perturb_graph(self.orig_graph.detach().clone(), self.perb_list[i].get_node_pairs())
            print ("Targets:", self.targets[i], self.target_nodes[i])
            print ("Perturbations:", self.perb_list[i].get_node_pairs())
            print (i, perb_graph.shape)
            print ()
