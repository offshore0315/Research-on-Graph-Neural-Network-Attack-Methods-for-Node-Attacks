from torch_geometric.nn import Node2Vec
import torch
from sklearn.linear_model import LogisticRegression
import pickle
import os
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

class N2V (torch.nn.Module):
    def __init__(self, edge_index, embedding_dim, walk_length, context_size,
                 walks_per_node=1, p=1, q=1, num_negative_samples=1, device='cpu',
                 num_nodes=None, sparse=True, down_task="node_classification", dataset=None, **kwargs):
        super(N2V, self).__init__()
        self.n2v = Node2Vec(edge_index, embedding_dim, walk_length, context_size,
                        walks_per_node=walks_per_node, p=p, q=q, num_negative_samples=num_negative_samples,
                        num_nodes=num_nodes, sparse=sparse)
        self.down_task = down_task
        self.device = device
        self.dataset = dataset
        if (down_task == "node_classification"):
            file_dir = os.path.dirname(__file__)
            if (os.path.exists(f"{file_dir}/{self.down_task}/{self.dataset}/model-Node2Vec-logreg.pkl")):
                self.logreg = pickle.load(open (f"{file_dir}/{self.down_task}/{self.dataset}/model-Node2Vec-logreg.pkl", "rb"))
            else:
                self.logreg = LogisticRegression(solver='lbfgs', multi_class='auto')

    def pos_sample(self, N, edge_index, batch):
        batch = batch.repeat(self.n2v.walks_per_node)
        row, col = edge_index
        rowptr, col, _ = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).csr()
        rw = random_walk(rowptr, col, batch, self.n2v.walk_length, self.n2v.p, self.n2v.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.n2v.walk_length + 1 - self.n2v.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.n2v.context_size])
        return torch.cat(walks, dim=0)


    def neg_sample(self, N, edge_index, batch):
        batch = batch.repeat(self.n2v.walks_per_node * self.n2v.num_negative_samples)
        rw = torch.randint(N, (batch.size(0), self.n2v.walk_length))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.n2v.walk_length + 1 - self.n2v.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.n2v.context_size])
        return torch.cat(walks, dim=0)

    def forward (self):
        return self.n2v()
       
    def predict(self, z, classify=False, to_flatten=True):
        z = z.unsqueeze(0) if (z.dim() == 1) else z
        if (self.down_task == "node_classification"):
            return torch.tensor(self.logreg.predict(z.detach().numpy()))
        elif (self.down_task in ["link_prediction", "link_pair"]):
            z = z.unsqueeze(-1)
            out = torch.matmul(z[0, :].transpose(-1,-2), z[1, :]).squeeze()
            out = torch.sigmoid (out)
            if (classify):
                return torch.where(out > 0.5, 1, 0)
        return out

    def fit (self, train_idx, train_y, val_idx, val_y, n_epochs=100):
        loader = self.n2v.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(self.n2v.parameters()), lr=0.01)
        def train():
            self.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.n2v.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        for epoch in range(1, n_epochs+1):
            loss = train()
            self.n2v.eval()
            z = self.n2v()
            acc = self.n2v.test(z[train_idx], train_y, z[val_idx], val_y)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
        
        self.logreg = LogisticRegression(solver='lbfgs', multi_class='auto')
        self.logreg.fit (z[train_idx].detach().numpy(), train_y.detach().numpy())
        pickle.dump(self.logreg, open (f"{self.down_task}/{self.dataset}/model-Node2Vec-logreg.pkl", "wb"))

    @torch.no_grad()
    def test(self, idx, y):
        self.eval()
        z = self.n2v()
        return self.logreg.score(z[idx].detach().numpy(), y.detach().numpy())