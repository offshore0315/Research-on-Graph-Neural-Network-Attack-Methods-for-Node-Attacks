from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
import pickle
import torch.nn.functional as F

from cmd_args import cmd_args
import sys
sys.path.append('..')
import utils 
from grl_models import AttackModel
from torch.utils.data import DataLoader
import random
import numpy as np
from collections import deque
import os
from copy import deepcopy 

root = os.getcwd()
if (cmd_args.dataset in ["cora", "citeseer", "pubmed"]):
    data_root = root + "/../data/Planetoid/"
    data = Planetoid(root=data_root, name=cmd_args.dataset.capitalize(), transform=NormalizeFeatures())[0]
else:
    data_root = root + "/../data/"
    data = utils.load_other_datasets(data_root)

# print (data)
"""
if (cmd_args.directed == 0):
    from torch_geometric.utils import to_undirected
    print(data.edge_index.shape)
    data.edge_index = to_undirected(data.edge_index)
    print(data.edge_index.shape)
"""
if (cmd_args.lcc):
    data = utils.get_largest_component(data)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if (cmd_args.model_name == "CorrectAndSmooth"):
    # data = utils.get_largest_component(data)
    data = utils.preprocess(data, data_root, cmd_args.down_task, train_p=0.4, val_p=0.2, test_p=0.4)
    print (data)
elif (cmd_args.model_name == "PGNN"):
    data = utils.preprocess(data, data_root, cmd_args.down_task, train_p=0.5, val_p=0.1, test_p=0.1)
else:
    data = utils.preprocess(data, data_root, cmd_args.down_task)
# print (data)

idx_file_end = cmd_args.down_task + "_" + str(cmd_args.seed) + ".pt"
if (cmd_args.lcc):
    idx_file_end = 'lcc_' + idx_file_end

try:
    print (torch.all(data.train_mask == torch.load(data_root + cmd_args.dataset.capitalize() + "/train_" + idx_file_end)))
    print (torch.all(data.val_mask == torch.load(data_root + cmd_args.dataset.capitalize() + "/val_" + idx_file_end)))
    print (torch.all(data.test_mask == torch.load(data_root + cmd_args.dataset.capitalize() + "/test_" + idx_file_end)))
except:
    pass

data = data.to(cmd_args.device)

model_config = {
    "model_name": cmd_args.model_name,
    "layer_name": cmd_args.layer.upper(), 
    "dataset": cmd_args.dataset,
    "down_task": cmd_args.down_task,
    "dropout": cmd_args.dropout_rate, 
    "hidden_layers": cmd_args.hidden_layers, 
    "input": data.x.shape[1], 
    "output": torch.unique(data.train_y).shape[0],
    "device": cmd_args.device,
    "lr": cmd_args.lr,
    "n_nodes": data.x.shape[0], 
    # "feat_dim": 32,
}

if (cmd_args.model_name == "Node2Vec"):
    model_config["edge_index"] = data.edge_index

model = AttackModel(model_config)
# model = torch.nn.DataParallel(model)
print (model)
model = model.to(cmd_args.device)
model.eval()

model_save_path = model_config["down_task"] + "/" + model_config["dataset"] + "/" + cmd_args.model_save_name
# saving the characteristics
pickle.dump(model_config, open(model_save_path + "-args.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.lr, weight_decay=5e-4)

if (cmd_args.model_name in ["GNNGuard", "GNNNoGuard"]):
    from torch_geometric.utils import to_scipy_sparse_matrix
    from scipy.sparse import csr_matrix
    adj = csr_matrix(to_scipy_sparse_matrix(data.edge_index))
    features = csr_matrix(data.x.cpu())
    labels = data.y.cpu().numpy()
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask
    # from deeprobust.graph.data import Dataset
    # drg_data = Dataset(root=data_root, name=cmd_args.dataset, setting="gcn")
    # adj, features, labels = drg_data.adj, drg_data.features, drg_data.labels
    # idx_train, idx_val, idx_test = drg_data.idx_train, drg_data.idx_val, drg_data.idx_test
    model.fit (features, adj, data.train_y, data.val_y, idx_train, idx_val, idx_test=idx_test, n_epochs=cmd_args.n_epochs)
    print(model.test (features, adj, idx_test, data.test_y)[0])
elif (cmd_args.model_name == "CorrectAndSmooth"):
    # from torch_geometric.transforms import ToSparseTensor
    # data = ToSparseTensor(data)
    # features = csr_matrix(data.x)
    adj = data.edge_index
    features = data.x
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask
    model.fit (features, adj, data.train_y, data.val_y, idx_train, idx_val, idx_test=idx_test, n_epochs=cmd_args.n_epochs)
    # print('Train', model.model.evalMLP(features[idx_train], data.train_y).item(), model.test (features, adj, idx_train, data.train_y))
    # print('Val', model.model.evalMLP(features[idx_val], data.val_y).item(), model.test (features, adj, idx_val, data.val_y))
    test_idx = torch.cat((idx_val, idx_test))
    test_y = torch.cat((data.val_y, data.test_y))
    print('Test', model.model.evalMLP(features[idx_test], data.test_y).item(), model.test (features, adj, idx_test, data.test_y))
    print('Test', model.model.evalMLP(features[test_idx], test_y).item(), model.test (features, adj, test_idx, test_y))
    torch.save(torch.cat((idx_val, idx_test)), f'{cmd_args.down_task}/{cmd_args.dataset}/CAS_test_idx.pt')
    torch.save(torch.cat((data.val_y, data.test_y)), f'{cmd_args.down_task}/{cmd_args.dataset}/CAS_test_y.pt')
elif (cmd_args.model_name == "Node2Vec"):
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask
    model.fit (data.x, data.edge_index, data.train_y, data.val_y, idx_train, idx_val, idx_test=idx_test, n_epochs=cmd_args.n_epochs)
    print(model.test (data.x, data.edge_index, idx_test, data.test_y))
elif (cmd_args.down_task == "unsupervised"):
    from model_utils import UnsupervisedSampler, unsup_loss, logistic_test
    sampler = UnsupervisedSampler(data.edge_index, walk_length=7, context_size=5, walks_per_node=10)
    loader = sampler.loader(batch_size=cmd_args.batch_size, shuffle=True, num_workers=4)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            pos_rw, neg_rw = pos_rw.to(cmd_args.device), neg_rw.to(cmd_args.device)
            optimizer.zero_grad()
            z = model(data.x, data.edge_index)
            loss = unsup_loss(z, pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def test():
        model.eval()
        z = model(data.x, data.edge_index)
        acc = logistic_test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    loss_queue = deque([], 4)
    for epoch in range(1, cmd_args.n_epochs+1):
        loss = train()
        if ((len(loss_queue) > 0) and (sum(list(loss_queue))/len(loss_queue) < loss)):
            break
        loss_queue.append(loss)
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
else:
    if (cmd_args.down_task in ["link_prediction", "link_pair"]):
        sup_loss = torch.nn.BCEWithLogitsLoss() if (cmd_args.model_name == "PGNN") else torch.nn.BCELoss()
    else:
        sup_loss = torch.nn.NLLLoss ()

    def sample_train (inds):
        return data.train_mask.T[inds].T, data.train_y[inds]
    
    loader = DataLoader(range(data.train_y.shape[0]), collate_fn = sample_train, batch_size=cmd_args.batch_size, shuffle=True, num_workers=(0 if ("cuda" in cmd_args.device) else 4)) 

    def train():
        model.train()
        total_loss = 0
        for batch_mask, batch_y in loader:
            batch_mask, batch_y = batch_mask, batch_y
            optimizer.zero_grad()
            z = model(data.x, data.edge_index)
            out = model.predict(z[batch_mask])
            loss = sup_loss(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
        pred = model.predict(z[data.train_mask], classify=True).detach()
        train_correct = pred == data.train_y  # Check against ground-truth labels.
        train_acc = int(train_correct.sum()) / int(len(train_correct))
        return total_loss / len(loader), train_acc

    def test():
        model.eval()
        z = model(data.x, data.edge_index)  # Perform a single forward pass.
        pred = model.predict(z[data.test_mask], classify=True)
        test_correct = pred == data.test_y  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(len(test_correct))  # Derive ratio of correct predictions.
        return test_acc

    best_loss_val, best_acc_val = 100, 0.0
    early_stopping, patience = cmd_args.patience, cmd_args.patience
    best_weights = None
    for epoch in range(1, cmd_args.n_epochs+1):
        train_loss, train_acc = train()
        z = model(data.x, data.edge_index)
        out = model.predict(z[data.val_mask])
        val_loss = sup_loss(out, data.val_y).detach().item()
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
        pred = model.predict(z[data.val_mask], classify=True)
        val_correct = pred == data.val_y  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / int(len(val_correct))
        # if (best_loss_val > val_loss):
        #     best_loss_val = val_loss
        #     best_weights = deepcopy(model.state_dict())
        if (val_acc > best_acc_val):
            best_acc_val = val_acc
            best_weights = deepcopy(model.state_dict())
            patience = early_stopping
        else:
            patience -= 1
        if ((epoch > early_stopping) and (patience <= 0)):
            break
        del z, out, pred

    if (patience > 0):
        model.load_state_dict(best_weights)
    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

# 
# exit()
torch.save(model.state_dict(), model_save_path + ".model")

emb_save_path = "emb_models/" + model_config["dataset"] + "/" + cmd_args.model_save_name

if (cmd_args.save_embs):
    import numpy as np
    orig_embeds = model(data.x, data.edge_index).detach().numpy()
    np.save(open(F"{emb_save_path}.npy", 'wb'), orig_embeds)
