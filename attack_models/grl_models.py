from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import ModuleList
from torch.nn import Identity 
from torch_sparse import SparseTensor
import os

import pickle

class DeepGRL (nn.Module):
    def __init__ (self, kwargs):
        super(DeepGRL, self).__init__()
        self.layers = ModuleList()
        # if (kwargs["batch_norm"]):
        self.batch_norms = ModuleList()
        for hid_size in kwargs['hidden_layers']:
            # if (kwargs["down_task"] == "link_prediction"):
            #     self.batch_norms.append(nn.BatchNorm2d(hid_size))
            # else:
            self.batch_norms.append(nn.BatchNorm1d(hid_size))
        if (kwargs['layer_name'] == 'GCN'):
            from torch_geometric.nn import GCNConv
            in_size = kwargs['input']
            for out_size in kwargs['hidden_layers']:
                self.layers.append(GCNConv(in_size, out_size))
                in_size = out_size
            #self.layers.append(nn.Linear(in_size, kwargs['output']))
            # self.layers.append(GCNConv(in_size, kwargs['output']))
        elif (kwargs['layer_name'] == 'GAT'):
            from torch_geometric.nn import GATConv
            in_size = kwargs['input']
            for out_size in kwargs['hidden_layers']:
                self.layers.append(GATConv(in_size, out_size))
                in_size = out_size
            #self.layers.append(nn.Linear(in_size, kwargs['output']))
            # self.layers.append(GATConv(in_size, kwargs['output']))
            #pass
        elif (kwargs['layer_name'] == 'SAGE'):
            from torch_geometric.nn import SAGEConv
            in_size = kwargs['input']
            for out_size in kwargs['hidden_layers']:
                self.layers.append(SAGEConv(in_size, out_size))
                in_size = out_size
            #self.layers.append(nn.Linear(in_size, kwargs['output']))
            # self.layers.append(SAGEConv(in_size, kwargs['output']))
            #pass
        elif (kwargs['layer_name'] == 'GIN'):
            from torch_geometric.nn import GINConv
            in_size = kwargs['input']
            for out_size in kwargs['hidden_layers']:
                self.layers.append(GINConv(nn.Sequential(nn.Linear(in_size, 16), nn.Linear(16, out_size))))
                in_size = out_size
            #self.layers.append(nn.Linear(in_size, kwargs['output']))
            # self.layers.append(GINConv(in_size, kwargs['output']))
            #pass
        elif (kwargs['layer_name'] == 'N2V'):
            pass
        # it should be a linear layer according to me:
        # final layer specific to downstream task
        if (kwargs['output'] is not None):
            if (kwargs['down_task'] == "node_classification"):
                self.final_layer = nn.Linear(in_size, kwargs['output'])
            elif (kwargs['down_task'] in ["link_prediction", "link_pair"]):
                self.final_layer = nn.Linear(in_size, in_size)#kwargs['output'])
        self.dropout_rate = kwargs['dropout']
        self.down_task = kwargs['down_task']
        self.layer_name = kwargs['layer_name']

    def forward(self, x, edge_index, neg_edge_index=None, predict=False, to_flatten=True):
        # would always return embeddings
        for n_layer, graph_deep_layer in enumerate(self.layers[:-1]):
            x = graph_deep_layer(x, edge_index)
            # if (kwargs["batch_norm"]):
            x = self.batch_norms[n_layer](x)
            if (self.layer_name != "GIN"):
                x = x.relu()
            x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.layers[-1](x, edge_index)
        # if (kwargs["batch_norm"]):
        # x = self.batch_norms[-1](x)
        if (predict):
            return self.predict (x, edge_index, classify=True, to_flatten=to_flatten)
        return x

    def predict (self, z, classify=False, to_flatten=True):
        # doubt about these two --
        z = z.relu()
        z = F.dropout(z, self.dropout_rate, training=self.training)
        if (self.down_task == "node_classification"):
            out = self.final_layer(z)
            out = F.log_softmax(out, dim=-1)
            if (classify):
                return out.argmax(dim=-1)
        elif (self.down_task in ["link_prediction", "link_pair"]):
            # out.shape = (2, n_edges, 2)
            #z = torch.cat((z[0, :], z[1, :]), dim=1)
            #out = self.final_layer(z)
            # z = self.final_layer(z)
            # z = F.softmax(z, dim=-1).unsqueeze(3)
            ndims = len(z.shape)
            z = z.unsqueeze(-1)
            out = torch.matmul(z[0, :].transpose(-1,-2), z[1, :]).squeeze()
            out = torch.sigmoid (out)
            if (classify):
                return torch.where(out > 0.5, 1, 0)
        return out

class AttackModel (nn.Module):
    def __init__(self, kwargs):
        super(AttackModel, self).__init__()
        # multiple layers not implemented yet but can be done 
        # need to have the exact sizes of each layer in kwargs['hidden_layers']
        self.model_name = kwargs['model_name']
        self.dropout_rate = kwargs['dropout']
        self.down_task = kwargs['down_task']
        if (kwargs['dataset'] in ["cora", "citeseer", "pubmed"]):
            self.data_dir = f"data/Planetoid/{kwargs['dataset'].capitalize()}"
        else:
            self.data_dir = f"data/{kwargs['dataset']}"
        if (kwargs['model_name'] == "Deep"):
            self.model = DeepGRL(kwargs)
        # elif (kwargs['model_name'] == "GIN"):
        #     from torch_geometric.nn import GINConv
        #     self.model = 
        elif (kwargs['model_name'] == "Node2Vec"):
            try:
                from n2v_model import N2V
            except:
                from attack_models.n2v_model import N2V
            self.model = N2V(kwargs["edge_index"], embedding_dim=kwargs['hidden_layers'][-1], 
                            walk_length=20, context_size=10, walks_per_node=10,
                            num_negative_samples=1, p=1, q=1, sparse=True, device=kwargs["device"], 
                            down_task=kwargs["down_task"], dataset=kwargs["dataset"])
        elif (kwargs['model_name'] == "GNNGuard"):
            if (kwargs['layer_name'] == 'GCN'):
                try:
                    from gcn_guard import GCN_Guard
                except:
                    from attack_models.gcn_guard import GCN_Guard
                self.model = GCN_Guard(kwargs["input"], kwargs["hidden_layers"][0], kwargs["output"], dropout=kwargs["dropout"], 
                                        lr=kwargs["lr"], down_task=kwargs["down_task"], device=kwargs['device'])
        elif (kwargs['model_name'] == "GNNNoGuard"):
            if (kwargs['layer_name'] == 'GCN'):
                try:
                    from gcn_guard import GCN_Guard
                except:
                    from attack_models.gcn_guard import GCN_Guard
                self.model = GCN_Guard(kwargs["input"], kwargs["hidden_layers"][0], kwargs["output"], dropout=kwargs["dropout"], 
                                        lr=kwargs["lr"], down_task=kwargs["down_task"], device=kwargs['device'], attention=False)
        elif (kwargs['model_name'] == "CorrectAndSmooth"):
            try:
                from correct_and_smooth import CAS_NC
            except:
                from attack_models.correct_and_smooth import CAS_NC
            self.model = CAS_NC(kwargs["input"], kwargs["output"], dropout=kwargs["dropout"], device=kwargs["device"],
                            num_correction_layers=100, correction_alpha=0.6, num_smoothing_layers=100, 
                            smoothing_alpha=0.6, autoscale=True, scale=100., train_only=True,
                            dataset=kwargs["dataset"], down_task=kwargs["down_task"])
        elif (kwargs['model_name'] == "PGNN"):
            try:
                from pgnn_model import PGNN
            except:
                from attack_models.pgnn_model import PGNN
            self.dataset = kwargs["dataset"]
            self.model = PGNN(input_dim=kwargs['input'], feature_dim=kwargs["feat_dim"] if ("feat_dim" in kwargs) else 256,
                            hidden_dim=kwargs["hidden_layers"][0], output_dim=kwargs["hidden_layers"][-1],
                            n_classes=kwargs["output"], feature_pre=True, layer_num=len(kwargs['hidden_layers']), 
                            dropout=kwargs['dropout'], down_task=kwargs["down_task"], device=kwargs['device'],
                            n_nodes=None, dataset=kwargs["dataset"])

    def forward(self, x, edge_index, predict=False, to_flatten=True):
        if (self.model_name == "Deep"):
            return self.model(x, edge_index, predict=predict, to_flatten=to_flatten)
        elif (self.model_name in ["GNNGuard", "GNNNoGuard"]):
            return self.model(x, edge_index, sparse_input=False)
        elif (self.model_name == "CorrectAndSmooth"):
            (row, col), N = edge_index, x.shape[0]
            adj_t = SparseTensor(row=col, col=row, value=None,
                            sparse_sizes=(N, N), is_sorted=True)
            return self.model(x, adj_t)
        elif (self.model_name == "PGNN"):
            try:
                from pgnn_utils import preprocess, preselect_anchor, precompute_dist_data
            except:
                from attack_models.pgnn_utils import preprocess, preselect_anchor, precompute_dist_data
            file_dir = os.path.dirname(__file__)
            if (self.training):
                dists = preprocess(edge_index, x.shape[0], data_dir=self.data_dir, use_saved=True)
                dists_max, dists_argmax= preselect_anchor(x.shape[0], dists)
                # if (os.path.exists(f"{file_dir}/../{self.data_dir}/dists_max.pt")):
                #     dists_max = torch.load(f"{file_dir}/../{self.data_dir}/dists_max.pt")
                #     dists_argmax = torch.load(f"{file_dir}/../{self.data_dir}/dists_argmax.pt")
                # else:
                #     dists = preprocess(edge_index, x.shape[0])
                #     dists_max, dists_argmax= preselect_anchor(x.shape[0], dists)
                #     torch.save(dists_max, f"{file_dir}/../{self.data_dir}/dists_max.pt")
                #     torch.save(dists_argmax, f"{file_dir}/../{self.data_dir}/dists_argmax.pt")
            else:
                # dists = preprocess(edge_index, x.shape[0]).to(self.model.device)
                dists = precompute_dist_data(edge_index.cpu().numpy(), x.shape[0], approximate=-1)
                dists = torch.from_numpy(dists).float()
                # print ("preselecting started")
                if (self.dataset == "pubmed"):
                    dists_max, dists_argmax= preselect_anchor(x.shape[0], dists, c=0.1, device=self.model.device)
                else:
                    dists_max, dists_argmax= preselect_anchor(x.shape[0], dists, c=1, device=self.model.device)
                # print ("preselecting finished")
                del dists
            return self.model.forward(x, dists_max, dists_argmax, predict=predict)
        elif (self.model_name == "Node2Vec"):
            return self.model()
        else:
            return self.model(x, edge_index)

    def predict (self, z, classify=False, to_flatten=True):
        return self.model.predict(z, classify=classify, to_flatten=to_flatten)

    def fit (self, features, adj, train_labels, val_labels, idx_train, idx_val, idx_test=None, n_epochs=81, patience=510):
        if (self.model_name == "GNNGuard"):
            return self.model.fit (features, adj, train_labels, val_labels, idx_train, idx_val=idx_val, idx_test=idx_test,
                            train_iters=n_epochs, attention=True, patience=patience, verbose=True)
        elif (self.model_name == "GNNNoGuard"):
            return self.model.fit (features, adj, train_labels, val_labels, idx_train, idx_val=idx_val, idx_test=idx_test,
                            train_iters=n_epochs, attention=False, patience=patience, verbose=True)
        elif (self.model_name == "CorrectAndSmooth"):
            return self.model.fit_mlp (features, train_labels, val_labels, idx_train, idx_val, n_epochs=n_epochs)
        elif (self.model_name == "Node2Vec"):
            return self.model.fit (idx_train, train_labels, idx_val, val_labels, n_epochs=n_epochs)
        else:
            # will not be used --
            return self.model.fit (features, adj, labels, idx_train)

    def test (self, x, edge_index, idx_test, test_labels):
        if (self.model_name in ["GNNGuard", "GNNNoGuard"]):
            return self.model.test (idx_test, test_labels)
        elif (self.model_name == "CorrectAndSmooth"):
            (row, col), N = edge_index, x.shape[0]
            adj_t = SparseTensor(row=col, col=row, value=None,
                            sparse_sizes=(N, N), is_sorted=True)
            return self.model.test (x, adj_t, idx_test, test_labels)
        else:
            return self.model.test (idx_test, test_labels)
