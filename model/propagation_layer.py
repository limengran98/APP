import torch
import random
from torch_geometric.typing import Adj
from torch_scatter import scatter_mean

from model.graph_tools import (
    get_propagation_matrix,
    get_edge_index_from_y,
    get_edge_index_from_y_ratio,
    LARGE_GRAPH_THRESHOLD
)

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
            # Optimization: Skip for large datasets
            if self.n_nodes > LARGE_GRAPH_THRESHOLD:
                self._label_mask_25 = torch.tensor([], dtype=torch.long)
                self._label_adj_25 = get_propagation_matrix(torch.empty((2,0), dtype=torch.long), self.n_nodes)
            else:
                _, label_mask_50 = self.label_adj_50()
                self._label_mask_50 = torch.tensor(random.sample(label_mask_50.tolist(), int(0.5*label_mask_50.size(0))),dtype=torch.long)
                edge_index = get_edge_index_from_y(self.y, self._label_mask_25)
                self._label_adj_25 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_25, self._label_mask_25

    def label_adj_50(self):
        if self._label_adj_50 is None:
            # Optimization: Skip for large datasets
            if self.n_nodes > LARGE_GRAPH_THRESHOLD:
                self._label_mask_50 = torch.tensor([], dtype=torch.long)
                self._label_adj_50 = get_propagation_matrix(torch.empty((2,0), dtype=torch.long), self.n_nodes)
            else:
                _, label_mask_75 = self.label_adj_75()
                self._label_mask_75 = torch.tensor(random.sample(label_mask_75.tolist(), int(0.75*label_mask_75.size(0))),dtype=torch.long)
                edge_index = get_edge_index_from_y(self.y, self._label_mask_50)
                self._label_adj_50 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_50, self._label_mask_50

    def label_adj_75(self):
        if self._label_adj_75 is None:
            # Optimization: Skip for large datasets
            if self.n_nodes > LARGE_GRAPH_THRESHOLD:
                self._label_mask_75 = torch.tensor([], dtype=torch.long)
                self._label_adj_75 = get_propagation_matrix(torch.empty((2,0), dtype=torch.long), self.n_nodes)
            else:
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
        if out is None: out = self.out
        # Ensure device consistency
        device = out.device
        G = torch.ones(self.n_nodes, device=device)
        G[mask] = gamma
        G = G.unsqueeze(1)
        
        # Move other components to device
        mean = self.mean.to(device)
        std = self.std
        if isinstance(std, torch.Tensor): std = std.to(device)
        out_k_init = self.out_k_init.to(device)
        adj = adj.to(device)
        self_adj = self.adj.to(device)
        know_mask = self.know_mask.to(device)

        out = (out - mean) / std
        for _ in range(num_iter):
            out = G*(alpha*torch.spmm(self_adj, out)+(1-alpha)*out.mean(dim=0)) + (1-G)*torch.spmm(adj, out)
            out[know_mask] = beta*out[know_mask] + (1-beta)*out_k_init
        return out * std + mean

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
        if out is None: out = self.out
        
        # Device management
        device = out.device
        mean = self.mean.to(device)
        std = self.std
        if isinstance(std, torch.Tensor): std = std.to(device)
        out_k_init = self.out_k_init.to(device)
        self_adj = self.adj.to(device)
        know_mask = self.know_mask.to(device)
        
        out = (out - mean) / std
        
        # --- Core modification: Use scatter_mean instead of spmm(label_adj) for large datasets ---
        if self.n_nodes < LARGE_GRAPH_THRESHOLD:
            # Original logic: Use full connected matrix multiplication for small datasets
            label_adj_all = self.label_adj_all.to(device)
            for _ in range(num_iter):
                out = gamma*(alpha*torch.spmm(self_adj, out)+(1-alpha)*out.mean(dim=0)) + (1-gamma)*torch.spmm(label_adj_all, out)
                out[know_mask] = beta*out[know_mask] + (1-beta)*out_k_init
        else:
            # New logic: Use mean propagation for large datasets (Genius/Arxiv)
            y = self.y.to(device)
            for _ in range(num_iter):
                struct_term = alpha * torch.spmm(self_adj, out) + (1-alpha) * out.mean(dim=0)
                # Compute class centers
                class_means = scatter_mean(out, y, dim=0)
                # Broadcast back to nodes (equivalent to multiplying by normalized class adjacency matrix)
                label_term = class_means[y]
                out = gamma * struct_term + (1-gamma) * label_term
                out[know_mask] = beta * out[know_mask] + (1-beta) * out_k_init

        return out * std + mean
    
    def arb_label_all(self, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        if out is None: out = self.out
        
        device = out.device
        mean = self.mean.to(device)
        std = self.std
        if isinstance(std, torch.Tensor): std = std.to(device)
        out_k_init = self.out_k_init.to(device)
        self_adj = self.adj.to(device)
        know_mask = self.know_mask.to(device)

        out = (out - mean) / std
        
        # --- Core modification: Logic adaptation for large datasets ---
        if self.n_nodes < LARGE_GRAPH_THRESHOLD:
            label_adj_all = self.label_adj_all.to(device)
            for _ in range(num_iter):
                out = gamma*torch.spmm(self_adj, out) + (1-gamma)*torch.spmm(label_adj_all, out)
                out[know_mask] = beta*out[know_mask] + (1-beta)*out_k_init
        else:
            y = self.y.to(device)
            for _ in range(num_iter):
                struct_term = torch.spmm(self_adj, out)
                class_means = scatter_mean(out, y, dim=0)
                label_term = class_means[y]
                out = gamma * struct_term + (1-gamma) * label_term
                out[know_mask] = beta * out[know_mask] + (1-beta) * out_k_init

        return out * std + mean

    def PVP_label_100(self, out: torch.Tensor = None, K: torch.Tensor = None, alpha: float = 0.95, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        if out is None: out = self.out
        
        device = out.device
        mean = self.mean.to(device)
        std = self.std
        if isinstance(std, torch.Tensor): std = std.to(device)
        out_k_init = self.out_k_init.to(device)
        self_adj = self.adj.to(device)
        know_mask = self.know_mask.to(device)
        
        if isinstance(K, torch.Tensor): K = K.to(device)

        out = (out - mean) / std
        for _ in range(num_iter):
            out = gamma*(alpha*torch.spmm(self_adj, out)+(1-alpha)*out.mean(dim=0)) + (1-gamma)*torch.spmm(K, out)
            out[know_mask] = beta*out[know_mask] + (1-beta)*out_k_init
        return out * std + mean