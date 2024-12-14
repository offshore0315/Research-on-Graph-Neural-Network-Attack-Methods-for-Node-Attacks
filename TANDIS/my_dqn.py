import numpy as np
import random
from cmd_args import cmd_args
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from utils import perturb_graph

def bhop_nbrs (edge_index, node, b, include_direct=True):
    nodes = torch.tensor([node], dtype=torch.int64)
    neighbors = torch.tensor([], dtype=torch.int64)
    for i in range(b):
        if (nodes.numel() == 0):
            break
        inds = torch.searchsorted(edge_index[0], torch.stack((nodes, nodes + 1)))
        nodes1 = torch.cat([edge_index[1, inds[0, n]:inds[1, n]] for n in range(len(nodes))])
        inds = torch.searchsorted(edge_index[1], torch.stack((nodes, nodes + 1)))
        nodes2 = torch.cat([edge_index[0, inds[0, n]:inds[1, n]] for n in range(len(nodes))])
        nodes = torch.cat((nodes1, nodes2))
        if (include_direct or (not(include_direct) and (i > 0))):
            neighbors = torch.cat ((neighbors, nodes))
    return neighbors 

def pool_embeds (embeds, pool_type="max"):
    try:
        if (pool_type == "max"):
            pooled_embs = torch.max(embeds, 0)[0]
        elif (pool_type == "mean"):
            pooled_embs = torch.mean(embeds, 0)
        elif (pool_type == "sum"):
            pooled_embs = torch.mean(embeds, 0)
    except:
        pooled_embs = torch.rand(embeds.shape[1])
    # nan to zero
    pooled_embs[pooled_embs != pooled_embs] = 0
    return pooled_embs

def get_mu_s (mu_v, target_node, graph):
    # region1 = bhop_nbrs(graph, target_node, 1)
    # region2 = bhop_nbrs(graph, target_node, cmd_args.num_hops, include_direct=False)
    # mu_s = torch.cat ((pool_embeds (mu_v[region1, :]), pool_embeds (mu_v[region2, :])))
    region = bhop_nbrs(graph, target_node, cmd_args.num_hops)
    mu_s = pool_embeds (mu_v[region, :], pool_type="mean")
    mu_s = torch.cat ((mu_s, mu_v[target_node, :]))
    return mu_s

def get_mu_a (mu_v, actions):
    if torch.is_tensor(actions):
        return (torch.sign(actions).reshape((actions.shape[0], 1)) * mu_v[torch.abs(actions), :])
    else:
        # actions is a singleton integer -- np is really an overkill but works
        return (np.sign(actions) * mu_v[np.abs(actions), :])

class DQN (nn.Module):
    # consider only adding edges as actions ???
    # ?
    def __init__(self, orig_graph, hidden_size, node_features, candidates, mu_type="e2e_embeds", num_mu_levels=2, embed_dim=16):
        super(DQN, self).__init__()
        self.orig_graph = orig_graph
        self.mu_type = mu_type
        self.candidates  = candidates
        self.node_features = node_features
        self.directed = cmd_args.directed
        if (self.mu_type == "e2e_embeds"):
            # end-to-end training
            self.conv_layers = [GCNConv(node_features.shape[1], embed_dim)]
            self.num_mu_levels = num_mu_levels
            for i in range(self.num_mu_levels):
                self.conv_layers.append(GCNConv(embed_dim, embed_dim))
        # mu_s, mu_a
        self.input_size = 2*embed_dim + embed_dim
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)
    
    def get_all_actions (self, graph, t, sol_set):
        # adding and deleting - undirected:
        poss_nodes_mask = torch.ones(self.node_features.shape[0], dtype=bool)
        poss_nodes_mask[t] = False
        for j in range(sol_set.shape[0]):
            e = sol_set[j]
            v = e[0] if (e[1] == t) else e[1]
            poss_nodes_mask[v] = False
        return torch.where(poss_nodes_mask)[0]
        # only adding edges
        poss_nodes_mask = torch.ones(self.node_features.shape[0], dtype=bool)
        poss_nodes_mask[t] = False
        if (self.candidates):
            deselect_nodes = torch.ones(self.node_features.shape[0], dtype=bool)
            deselect_nodes[self.candidates[t]] = False
            poss_nodes_mask[deselect_nodes] = False
        # only adding edges - no deletion
        if (self.directed):
            n_s, n_e = torch.searchsorted(graph[0], torch.stack((t, t+1)))
            out_t_nodes = graph[1, n_s:n_e]
            # following is incorrect -
            n_s, n_e = torch.searchsorted(graph[1], torch.stack((t, t+1)))
            in_t_nodes = graph[0, n_s:n_e]
            poss_nodes_mask2 = poss_nodes_mask.detach().clone()
            poss_nodes_mask[out_t_nodes] = False
            poss_nodes_mask2[in_t_nodes] = False
            return torch.cat((torch.where(poss_nodes_mask)[0], - torch.where (poss_nodes_mask2)[0]))
        else:
            n_s, n_e = torch.searchsorted(graph[0], torch.stack((t, t+1)))
            conn_nodes = graph[1, n_s:n_e]
            poss_nodes_mask[conn_nodes] = False
            return torch.where(poss_nodes_mask)[0]
            # conn_nodes = torch.cat ((torch.tensor(t), in_t_nodes, out_t_nodes)).detach().numpy()
            # possible_nodes = torch.tensor(self.candidates[t]) if (self.candidates) else torch.tensor([i for i in range(self.node_features.shape[0]) if (i not in conn_nodes)])
            # n_nodes = possible_nodes.shape[0]
            # directions = torch.ones(2*n_nodes, dtype=torch.int32)
            # directions[n_nodes:] *= -1
            # possible_nodes = directions * possible_nodes.repeat(2)

    def node_greedy_actions(self, states, list_q):
        actions = []
        values = []
        target_nodes, batch_graphs = zip(*states)
        for i in range(len(batch_graphs)):
            edge_index = perturb_graph (self.orig_graph, batch_graphs[i].get_node_pairs())
            all_actions = self.get_all_actions(edge_index, target_nodes[i], batch_graphs[i].get_node_pairs())
            val, act_idx = torch.max(list_q[i], dim=0)
            assert (list_q[i].shape[0] == all_actions.shape[0])
            values.append(val)
            actions.append(all_actions[act_idx])
        return torch.cat(actions, dim=0), torch.cat(values, dim=0)


    def forward(self, states, actions, greedy_acts=False):
        # graphs are edge index which could be sparse as well
        # edge list basically 
        target_nodes, batch_graphs = zip(*states)
        list_q_pred = []
        for i in range(len(batch_graphs)):
            # can be parallelized - not really since parameters of NN are shared
            edge_index = perturb_graph (self.orig_graph, batch_graphs[i].get_node_pairs())
            mu_v = None
            if (self.mu_type == "e2e_embeds"):
                mu_v = self.node_features
                for j in range(self.num_mu_levels):
                    mu_v = self.conv_layers[j](mu_v, edge_index)
            elif (self.mu_type == "preT_embeds"):
                # find mu_si, mu_ai 
                # could have pre-trained embeddings
                assert cmd_args.embeds is not None
                mu_v = torch.from_numpy(np.load(cmd_args.embeds)).to(cmd_args.device)
            elif (self.mu_type == "hand_crafted"):
                # 
                pass
            else:
                raise AssertionError
            mu_si = get_mu_s (mu_v, target_nodes[i], edge_index)
            if (actions is None):
                # consider all possible actions 
                # evaluation
                #
                all_actions = self.get_all_actions(edge_index, target_nodes[i], batch_graphs[i].get_node_pairs())
                #print (i, edge_index.shape, target_nodes[i], all_actions.shape)
                """
                print (target_nodes[i])
                for act in all_actions:
                    perb_graph = perturb_graph (edge_index, torch.tensor([[target_nodes[i], act]]).long() if (act > 0) else torch.tensor([[-act, target_nodes[i]]]).long())
                    print (act, edge_index.shape, perb_graph.shape)
                print()
                """
                mu_ai = get_mu_a (mu_v, all_actions)
                mu_si = mu_si.repeat(mu_ai.shape[0], 1)
                embed_s_a = torch.cat((mu_si, mu_ai), dim=1)
            else:
                # for training
                mu_ai = get_mu_a (mu_v, actions[i])
                embed_s_a = torch.cat((mu_si, mu_ai), dim=0)
            embed_s_a = F.relu( self.fc1 (embed_s_a) )
            raw_pred = self.fc2 (embed_s_a)
            list_q_pred.append(raw_pred)

        if greedy_acts:
            actions, list_q_pred = self.node_greedy_actions(states, list_q_pred)

        return actions, list_q_pred


class TStepDQN (nn.Module):
    def __init__(self, num_steps, orig_graph, mu_type, node_features, candidates, num_mu_levels=2, embed_dim=16):
        super(TStepDQN, self).__init__()
        list_dqns = []

        for i in range(0, num_steps):
            list_dqns.append(DQN(orig_graph, cmd_args.dqn_hidden, node_features, candidates, mu_type=mu_type, num_mu_levels=num_mu_levels, embed_dim=embed_dim))

        self.list_dqns = nn.ModuleList(list_dqns)
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts=False):
        assert time_t >= 0 and time_t < self.num_steps

        return self.list_dqns[time_t](states, actions, greedy_acts=greedy_acts)
        
        
