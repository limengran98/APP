import torch
from torch import nn
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian

from utils.graph_ops import get_propagation_matrix

class FeaturePropagation(nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x: torch.Tensor, edge_index: Adj, mask: torch.Tensor) -> torch.Tensor:
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]
        else:
            out = x.clone()

        n_nodes = x.shape[0]
        adj = get_propagation_matrix(edge_index, n_nodes)
        for _ in range(self.num_iterations):
            out = torch.spmm(adj, out)
            out[mask] = x[mask]
        return out

class APA:
    def __init__(self, edge_index: Adj, x: torch.Tensor, know_mask: torch.Tensor, is_binary=True):
        self.edge_index = edge_index
        self.x = x
        self.n_nodes = x.size(0)
        self.know_mask = know_mask
        self.mean = 0 if is_binary else x[know_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (self.x[self.know_mask] - self.mean) / self.std
        self.out = torch.zeros_like(self.x)
        self.out[self.know_mask] = self.x[self.know_mask]
        self._adj = None

    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    def fp(self, out: torch.Tensor = None, num_iter: int = 30, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.know_mask] = self.out_k_init
        return out #* self.std + self.mean

    def fp_analytical_solution(self, **kw) -> torch.Tensor:
        adj = self.adj.to_dense()

        assert self.know_mask.dtype == torch.int64
        know_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        know_mask[self.know_mask] = True
        unknow_mask = torch.ones(self.n_nodes, dtype=torch.bool)
        unknow_mask[self.know_mask] = False

        A_uu = adj[unknow_mask][:, unknow_mask]
        A_uk = adj[unknow_mask][:, know_mask]

        L_uu = torch.eye(unknow_mask.sum()) - A_uu
        L_inv = torch.linalg.inv(L_uu)

        out = self.out.clone()
        out[unknow_mask] = torch.mm(torch.mm(L_inv, A_uk), self.out_k_init)

        return out * self.std + self.mean

    def pr(self, out: torch.Tensor = None, alpha: float = 0.55, num_iter: int = 30, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)
            out[self.know_mask] = self.out_k_init
        return out * self.std + self.mean

    def ppr(self, out: torch.Tensor = None, alpha: float = 0.85, weight: torch.Tensor = None, num_iter: int = 50,
            **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        if weight is None:
            weight = self.mean
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * weight
            out[self.know_mask] = self.out_k_init
        return out * self.std + self.mean

    def mtp(self, out: torch.Tensor = None, beta: float = 0.85, num_iter: int = 50, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def mtp_analytical_solution(self, beta: float = 0.85, **kw) -> torch.Tensor:
        n_nodes = self.n_nodes
        eta = (1 / beta - 1)
        edge_index, edge_weight = get_laplacian(self.edge_index, num_nodes=n_nodes, normalization="sym")
        L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to_dense()
        Ik_diag = torch.zeros(n_nodes)
        Ik_diag[self.know_mask] = 1
        Ik = torch.diag(Ik_diag)
        out = (self.out - self.mean) / self.std
        out = torch.mm(torch.inverse(L + eta * Ik), eta * torch.mm(Ik, out))
        return out * self.std + self.mean

    def umtp(self, out: torch.Tensor = None, alpha: float = 0.9, beta: float = 0.5, num_iter: int = 2,
             **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def umtp2(self, out: torch.Tensor = None, alpha: float = 0.9, beta: float = 0.2, gamma: float = 0.75,
              num_iter: int = 20, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma * (alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)) + (1 - gamma) * out
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def umtp_analytical_solution(self, alpha: float = 0.85, beta: float = 0.70, **kw) -> torch.Tensor:
        n_nodes = self.n_nodes
        theta = (1 - 1 / self.n_nodes) * (1 / alpha - 1)
        eta = (1 / beta - 1) / alpha
        edge_index, edge_weight = get_laplacian(self.edge_index, num_nodes=n_nodes, normalization="sym")
        L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to_dense()
        Ik_diag = torch.zeros(n_nodes)
        Ik_diag[self.know_mask] = 1
        Ik = torch.diag(Ik_diag)
        L1 = torch.eye(n_nodes) * (n_nodes / (n_nodes - 1)) - torch.ones(n_nodes, n_nodes) / (n_nodes - 1)
        out = (self.out - self.mean) / self.std
        out = torch.mm(torch.inverse(L + eta * Ik + theta * L1), eta * torch.mm(Ik, out))
        return out * self.std + self.mean