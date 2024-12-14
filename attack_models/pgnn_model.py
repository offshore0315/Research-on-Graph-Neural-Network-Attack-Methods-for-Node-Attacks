import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init
import numpy as np

### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

# # PGNN layer, only pick closest node for message passing
class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()
        messages = feature[dists_argmax.flatten(), :]
        messages = messages.reshape((dists_argmax.shape[0], dists_argmax.shape[1], feature.shape[1]))
        messages = messages * dists_max.unsqueeze(-1)
        
        # print ("here1")
        feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        # print ("here2", feature.shape, messages.shape)
        messages = torch.cat((messages, feature), dim=-1)
        # print ("here3", messages.shape)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages) # n*m*d
        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure

class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, n_classes, feature_pre=True, 
                layer_num=2, dropout=True, down_task='node_classification', device='cpu', n_nodes=None, dataset=None, **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.down_task = down_task
        self.device = device
        self.dataset = dataset
        self.feature_dim = feature_dim
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.n_nodes = n_nodes
            self.conv_out = PGNN_layer(hidden_dim, output_dim)
            if (down_task == "node_classification"):
                if (feature_pre):
                    self.conv_out = PGNN_layer(hidden_dim, n_classes)
                    self.out_fc = nn.Identity()
                else:
                    self.conv_out = PGNN_layer(hidden_dim, output_dim)
                    position_dim = int(np.log2(n_nodes))**2 if (n_nodes is not None) else output_dim
                    self.out_fc = nn.Linear (position_dim, n_classes)

    def forward(self, x, dists_max, dists_argmax, predict=False):
        x = x.to(self.device)
        dists_max, dists_argmax = dists_max.to(self.device), dists_argmax.to(self.device)
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, dists_max, dists_argmax)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, dists_max, dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, dists_max, dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        if ((self.down_task == "node_classification") and (self.n_nodes is None)):
            return self.predict(x) if (predict) else x
        else:
            return self.predict(x_position) if (predict) else x_position

    def predict (self, z, classify=False, to_flatten=True):
        if (self.down_task == "node_classification"):
            z = self.out_fc(z)
            out = F.log_softmax(z, dim=-1)
            if (classify):
                return out.argmax(dim=-1)
        elif (self.down_task in ["link_prediction", "link_pair"]):
            out = torch.sum(z[0, :] * z[1, :], dim=-1)
            out = torch.sigmoid(out)
            if (classify):
                if (self.dataset == "cora"):
                    return torch.where(out > 0.666, 1, 0)
                elif (self.dataset == "citeseer"):
                    if (self.down_task == "link_pair"):
                        return torch.where(out > 0.729239, 1, 0)
                    return torch.where(out > 0.725, 1, 0)
            # z = z.unsqueeze(-1)
            # out = torch.matmul(z[0, :].transpose(-1,-2), z[1, :]).squeeze()
            # out = torch.sigmoid (out)
            # if (classify):
            #     out = torch.sigmoid (out)
            #     return torch.where(out > 0.5, 1, 0)
        return out
