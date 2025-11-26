import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SGConv, TransformerConv, SAGEConv, DirGNNConv, GPSConv, GATv2Conv, ClusterGCNConv

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.ae = nn.Linear(output_dim, input_dim)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        re_x = self.ae(x)
        return x, re_x
    def re(self, x):
        re_x = self.ae(x)
        return re_x


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class AP(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_dim, args):
        super(AP, self).__init__()
        if args.MPNN =='Dir':
            self.conv1 = DirGNNConv(TransformerConv(num_features, hidden_channels))
            self.conv2 = DirGNNConv(TransformerConv(hidden_channels, output_dim))
        # if args.MPNN =='GAT':
        #     self.conv1 = DirGNNConv(GATConv(num_features, hidden_channels))
        #     self.conv2 = DirGNNConv(GATConv(hidden_channels, hidden_channels))
        # if args.MPNN =='GraphSAGE':
        #     self.conv1 = DirGNNConv(SAGEConv(num_features, hidden_channels))
        #     self.conv2 = DirGNNConv(SAGEConv(hidden_channels, hidden_channels))
        if args.MPNN =='MLP':
            self.conv1 = torch.nn.Linear(num_features, hidden_channels)
            self.conv2 = torch.nn.Linear(hidden_channels, output_dim)

        if args.MPNN =='GCN':
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, output_dim)
            
        if args.MPNN =='GAT':
            self.conv1 = GATConv(num_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, output_dim)
        if args.MPNN =='GraphSAGE':
            self.conv1 = SAGEConv(num_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, output_dim)
        if args.MPNN =='Transformer':
            self.conv1 = TransformerConv(num_features, hidden_channels)
            self.conv2 = TransformerConv(hidden_channels, output_dim)
        if args.MPNN =='GATv2':
            self.conv1 = GATv2Conv(num_features, hidden_channels)
            self.conv2 = GATv2Conv(hidden_channels, output_dim)
        if args.MPNN =='SGC':
            self.conv1 = SGConv(num_features, hidden_channels)
            self.conv2 = SGConv(hidden_channels, output_dim)
        if args.MPNN =='Cluster':
            self.conv1 = ClusterGCNConv(num_features, hidden_channels)
            self.conv2 = ClusterGCNConv(hidden_channels, output_dim)
        if args.MPNN =='GPS':
            self.conv1 = GPSConv(num_features, GCNConv(num_features, num_features), attn_type = 'performer')
            self.conv2 = GCNConv(num_features, output_dim)
        self.dropout = args.dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x