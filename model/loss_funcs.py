import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian

def to_dirichlet_loss(attrs, laplacian):
    return torch.bmm(attrs.t().unsqueeze(1), laplacian.matmul(attrs).t().unsqueeze(2)).view(-1).sum()

def contrastive_proto_loss(features, prototypes, temperature=0.1):
    sim_matrix = F.cosine_similarity(
        features.unsqueeze(1), prototypes.unsqueeze(0), dim=-1
    ) / (temperature + 1e-8)
    proto_assign = torch.argmax(sim_matrix, dim=1)
    targets = F.one_hot(proto_assign, num_classes=prototypes.size(0)).float()
    logits = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    loss = - (targets * logits).sum(dim=1).mean()
    return loss

class arbLoss(nn.Module):
    def __init__(self, edge_index: Adj, raw_x: torch.Tensor, know_mask: torch.Tensor, alpha, beta, device, is_binary=False, **kw):
        super().__init__()

        self.device = device
        num_nodes = raw_x.size(0)
        self.n_nodes = num_nodes
        num_attrs = raw_x.size(1)
        self.know_mask = know_mask.to(device) 

        self.mean = 0 if is_binary else raw_x[know_mask].mean(dim=0).to(device)
        self.std = 1  
        self.out_k_init = (raw_x[know_mask] - self.mean) / self.std

        edge_index, edge_weight = get_laplacian(edge_index, num_nodes=num_nodes, normalization="sym")
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        self.L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(num_nodes, num_nodes)).to_dense().to(device)
        self.avg_L = num_nodes / (num_nodes - 1) * torch.eye(num_nodes, device=device) - 1 / (num_nodes - 1) * torch.ones(num_nodes, num_nodes, device=device)
        
        self.x = nn.Parameter(torch.zeros(num_nodes, num_attrs, device=device))
        self.x.data[know_mask] = raw_x[know_mask].clone().detach().data.to(device)
        
        if alpha == 0: alpha = 0.00001
        if beta == 0: beta = 0.00001
        self.theta = (1 - 1 / num_nodes) * (1 / alpha - 1)
        self.eta = (1 / beta - 1) / alpha

    def get_loss(self, x):
        x = (x - self.mean) / self.std
        dirichlet_loss = to_dirichlet_loss(x, self.L)
        avg_loss = to_dirichlet_loss(x, self.avg_L)
        recon_loss = nn.functional.mse_loss(x[self.know_mask], self.out_k_init, reduction="sum")
        return -(dirichlet_loss + self.eta * recon_loss + self.theta * avg_loss)

    def forward(self):
        return self.get_loss(self.x)

    def get_out(self):
        return self.x