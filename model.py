import os
import random
from collections import defaultdict
from itertools import permutations
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.typing import Adj
from torch_geometric.utils import (
    get_laplacian,
    remove_self_loops
)
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score,
    f1_score
)


def get_propagation_matrix(edge_index: Adj, n_nodes: int, mode: str = "adj") -> torch.sparse.FloatTensor:
    edge_index, edge_weight = get_laplacian(edge_index, num_nodes=n_nodes, normalization="sym")
    if mode == "adj":
        edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
    adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
    return adj



def get_edge_index_from_y(y: torch.Tensor, know_mask: Optional[torch.Tensor] = None) -> Adj:
    # Determine node indices to be included in computation
    if know_mask is None:
        nodes_idx = torch.arange(y.size(0), device=y.device)
        labels = y
    else:
        nodes_idx = know_mask.to(y.device)
        labels = y[nodes_idx]
    
    # Sort nodes by label for faster grouping
    sorted_labels, sort_idx = torch.sort(labels)
    sorted_nodes = nodes_idx[sort_idx]
    
    # Find boundaries of each label group
    diff_mask = torch.cat([torch.tensor([True], device=y.device), 
                           sorted_labels[1:] != sorted_labels[:-1],
                           torch.tensor([True], device=y.device)])
    split_indices = torch.where(diff_mask)[0]

    edge_list = []
    for i in range(len(split_indices) - 1):
        start, end = split_indices[i], split_indices[i + 1]
        group = sorted_nodes[start:end]
        if len(group) < 2:
            continue
        
        # Vectorized generation of all possible edges within the group
        n = len(group)
        grid = group.repeat(n, 1)
        src = grid.flatten()
        dst = grid.T.flatten()
        mask = src != dst
        edge_list.append(torch.stack([src[mask], dst[mask]]))
    
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long, device=y.device)
    return torch.cat(edge_list, dim=1)

def get_edge_index_from_y_ratio(y: torch.Tensor, ratio: float = 1.0) -> torch.Tensor:
    n = y.size(0)
    mask = []
    nodes = defaultdict(list)
    for idx, label in random.sample(list(enumerate(y.numpy())), int(ratio*n)):
        mask.append(idx)
        nodes[label].append(idx)
    arr = []
    for v in nodes.values():
        arr += list(permutations(v, 2))
    return torch.tensor(arr, dtype=torch.long).T, torch.tensor(mask, dtype=torch.long)


def to_dirichlet_loss(attrs, laplacian):
    return torch.bmm(attrs.t().unsqueeze(1), laplacian.matmul(attrs).t().unsqueeze(2)).view(-1).sum()

class arbLabel:

    def __init__(self, edge_index: Adj, x: torch.Tensor, y: torch.Tensor, know_mask: torch.Tensor, is_binary = False):
        self.x = x
        self.y = y
        self.n_nodes = x.size(0)
        self.edge_index = edge_index
        self._adj = None

        self._label_adj = None
        self._label_adj_25 = None
        self._label_adj_50 = None
        self._label_adj_75 = None
        self._label_adj_all = None
        self._label_mask = know_mask
        self._label_mask_25 = None
        self._label_mask_50 = None
        self._label_mask_75 = None

        self.know_mask = know_mask
        self.mean = 0 if is_binary else self.x[self.know_mask].mean(dim=0)
        self.std = 1 #if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (self.x[self.know_mask]-self.mean) / self.std
        # init self.out without normalized
        self.out = torch.zeros_like(self.x)
        self.out[self.know_mask] = self.x[self.know_mask]

    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    def label_adj(self):
        if self._label_adj is None:
            edge_index = get_edge_index_from_y(self.y, self.know_mask)
            self._label_adj = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj, self._label_mask
    
    def label_adj_25(self):
        if self._label_adj_25 is None:
            _, label_mask_50 = self.label_adj_50()
            self._label_mask_25 = torch.tensor(random.sample(label_mask_50.tolist(), int(0.5*label_mask_50.size(0))),dtype=torch.long)
            edge_index = get_edge_index_from_y(self.y, self._label_mask_25)
            self._label_adj_25 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_25, self._label_mask_25

    def label_adj_50(self):
        if self._label_adj_50 is None:
            _, label_mask_75 = self.label_adj_75()
            self._label_mask_50 = torch.tensor(random.sample(label_mask_75.tolist(), int(0.75*label_mask_75.size(0))),dtype=torch.long)
            edge_index = get_edge_index_from_y(self.y, self._label_mask_50)
            self._label_adj_50 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_50, self._label_mask_50

    def label_adj_75(self):
        if self._label_adj_75 is None:
            edge_index, self._label_mask_75 = get_edge_index_from_y_ratio(self.y, 0.75)
            self._label_adj_75 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_75, self._label_mask_75

    @property
    def label_adj_all(self):
        if self._label_adj_all is None:
            edge_index = get_edge_index_from_y(self.y)
            self._label_adj_all = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_all

    def arb(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean

    def _arb_label(self, adj: Adj, mask:torch.Tensor, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1):
        G = torch.ones(self.n_nodes)
        G[mask] = gamma
        G = G.unsqueeze(1)
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = G*(alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)) + (1-G)*torch.spmm(adj, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean

    def arb_label_25(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_25()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label_50(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_50()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label_75(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_75()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label_100(self, out: torch.Tensor = None, alpha: float = 0.95, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma*(alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)) + (1-gamma)*torch.spmm(self.label_adj_all, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean
    
    def arb_label_all(self, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma*torch.spmm(self.adj, out) + (1-gamma)*torch.spmm(self.label_adj_all, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean
    def PVP_label_100(self, out: torch.Tensor = None, K: torch.Tensor = None, alpha: float = 0.95, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma*(alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)) + (1-gamma)*torch.spmm(K, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean


class arbLoss(nn.Module):
    def __init__(self, edge_index: Adj, raw_x: torch.Tensor, know_mask: torch.Tensor, alpha, beta, device, is_binary=False, **kw):
        super().__init__()

        self.device = device
        num_nodes = raw_x.size(0)
        self.n_nodes = num_nodes
        num_attrs = raw_x.size(1)
        self.know_mask = know_mask.to(device)  # Ensure mask is on the same device

        self.mean = 0 if is_binary else raw_x[know_mask].mean(dim=0).to(device)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (raw_x[know_mask] - self.mean) / self.std

        edge_index, edge_weight = get_laplacian(edge_index, num_nodes=num_nodes, normalization="sym")
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        self.L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(num_nodes, num_nodes)).to_dense().to(device)
        self.avg_L = num_nodes / (num_nodes - 1) * torch.eye(num_nodes, device=device) - 1 / (num_nodes - 1) * torch.ones(num_nodes, num_nodes, device=device)
        
        self.x = nn.Parameter(torch.zeros(num_nodes, num_attrs, device=device))
        self.x.data[know_mask] = raw_x[know_mask].clone().detach().data.to(device)
        
        # Default values for alpha and beta
        if alpha == 0:
            alpha = 0.00001
        if beta == 0:
            beta = 0.00001
        self.theta = (1 - 1 / num_nodes) * (1 / alpha - 1)
        self.eta = (1 / beta - 1) / alpha
        # print(alpha, beta, self.theta, self.eta)

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
    


def get_filename(args):
    return f"PVP_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_iter{args.num_iter}_missrate{args.missrate}_data{args.data}.pt"


def save_or_load_x_PVP(x_missing, propagation_model, args, device):
    """Cache or load propagated features"""
    if not os.path.exists('PVP'):
        os.makedirs('PVP')
    filename = get_filename(args)
    filepath = os.path.join('PVP', filename)
    if os.path.exists(filepath):
        print(f"File {filename} exists. Loading.")
        x_PVP = torch.load(filepath, map_location=device)  # Force load to current device
    else:
        print(f"File {filename} not found. Computing and saving.")
        x_PVP = propagation_model.arb_label_100(
            x_missing, alpha=args.alpha, beta=args.beta, gamma=args.gamma, num_iter=args.num_iter
        ).to(device)
        torch.save(x_PVP, filepath)
    return x_PVP.to(device)  # Ensure it's on the right device


def robust_pseudo_labels(x, y_all, edge_index, num_classes):
    """Robust pseudo-label generation with multiple clustering models"""
    cluster_models = [
        KMeans(n_clusters=num_classes),
        MiniBatchKMeans(n_clusters=num_classes),
        AgglomerativeClustering(n_clusters=num_classes),
    ]
    all_preds = []
    for model in cluster_models:
        pred = model.fit_predict(x.cpu().numpy())
        all_preds.append(torch.tensor(pred))
    # Majority vote
    consensus_labels = torch.mode(torch.stack(all_preds), dim=0).values
    return evaluate_and_return(consensus_labels.cpu().numpy(), y_all.cpu().numpy())


def evaluate_and_return(labels, y_true):
    """Evaluate clustering results and return metrics"""
    nmi = normalized_mutual_info_score(y_true, labels)
    ac = accuracy_score(y_true, labels)
    f1 = f1_score(y_true, labels, average='micro')
    ari = adjusted_rand_score(y_true, labels)
    print(f'AC: {ac:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}')
    return ac, nmi, ari, f1, labels


def contrastive_proto_loss(features, prototypes, temperature=0.1):
    """Prototype-based contrastive loss"""
    sim_matrix = F.cosine_similarity(
        features.unsqueeze(1), prototypes.unsqueeze(0), dim=-1
    ) / (temperature + 1e-8)
    proto_assign = torch.argmax(sim_matrix, dim=1)
    targets = F.one_hot(proto_assign, num_classes=prototypes.size(0)).float()
    logits = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    loss = - (targets * logits).sum(dim=1).mean()
    return loss


class ProtoAwarePropagation(nn.Module):
    """Prototype-aware message propagation module"""
    def __init__(self, input_dim, num_prototypes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        attn_in_dim = input_dim * 2
        self.prototype_attn = nn.Sequential(
            nn.Linear(attn_in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.gate = nn.Linear(input_dim*3, 1)
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.beta = nn.Parameter(torch.tensor(1.))
        
    def forward(self, x, edge_index):
        row, col = edge_index
        proto_sim = F.cosine_similarity(x.unsqueeze(1), self.prototypes, dim=-1)
        proto_weights = F.softmax(proto_sim, dim=1)
        delta = x[row] - x[col]
        edge_weights = self.prototype_attn(torch.cat([x[row], delta], dim=1))
        same_proto = (proto_weights[row].argmax(1) == proto_weights[col].argmax(1)).float()
        edge_weights = edge_weights * same_proto.unsqueeze(1)
        propagated = scatter_mean(x[col] * edge_weights, row, dim=0, dim_size=x.size(0))
        gate_input = torch.cat([x, propagated, x-propagated], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        output = self.alpha * gate * propagated + self.beta * x
        return output + x


class EfficientPseudoLabel(nn.Module):
    """Memory-efficient pseudo-label generator"""
    def __init__(self, input_dim, num_prototypes):
        super().__init__()
        self.prototype_learner = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_prototypes)
        )
        self.topk = 20  # number of neighbors to sample
        
    def forward(self, x, edge_index):
        row, col = edge_index
        sampled = torch.randint(0, len(row), (self.topk,))
        proto_logits = self.prototype_learner(x)
        smooth_proto = scatter_mean(proto_logits[col[sampled]], row[sampled], dim=0)
        pseudo_labels = smooth_proto.argmax(dim=1)
        confidence = F.softmax(proto_logits, dim=1).max(dim=1)[0]
        return pseudo_labels, confidence



class HeteroImputation(nn.Module):
    """Feature imputation for heterophilic graphs"""
    def __init__(self, input_dim, num_prototypes):
        super().__init__()
        self.proto_prop = ProtoAwarePropagation(input_dim, num_prototypes)
        self.label_gen = EfficientPseudoLabel(input_dim, num_prototypes)
        
    def forward(self, x_missing, edge_index, num_iter=1):
        x_filled = x_missing.clone()
        for _ in range(num_iter):
            pseudo_labels, confidence = self.label_gen(x_filled, edge_index)
            prototype_matrix = self.proto_prop(x_filled, edge_index)
            prototype_matrix = confidence.unsqueeze(1) * prototype_matrix + \
                               (1-confidence.unsqueeze(1)) * prototype_matrix.detach()
        return prototype_matrix, pseudo_labels

class AdaptiveFeatureFusion(nn.Module):
    """Adaptive feature fusion"""
    def __init__(self, input_dim):
        super().__init__()
        
    def forward(self, x_filled, x_PVP, a, b):
        fused = a * x_filled + b * x_PVP
        fused += 0.01 * torch.tanh(x_filled * x_PVP)
        return fused
