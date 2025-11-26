import torch
import random
from collections import defaultdict
from itertools import permutations
from typing import Optional
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian, remove_self_loops

# --- Threshold definition: Exceeding this node count enables optimization for large datasets ---
LARGE_GRAPH_THRESHOLD = 50000

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
    
    # --- Optimization: Disable full connected label graph generation for large datasets ---
    if len(nodes_idx) > LARGE_GRAPH_THRESHOLD:
        return torch.empty((2, 0), dtype=torch.long, device=y.device)

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
        
        # Extra protection: Do not generate if a single class is too large, even if the total graph is small
        n = len(group)
        if n > 10000: 
            continue

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
    # --- Optimization: Directly return empty edges and random indices for large datasets ---
    if n > LARGE_GRAPH_THRESHOLD:
        indices = torch.randperm(n)[:int(ratio*n)]
        return torch.empty((2, 0), dtype=torch.long), indices

    mask = []
    nodes = defaultdict(list)
    for idx, label in random.sample(list(enumerate(y.numpy())), int(ratio*n)):
        mask.append(idx)
        nodes[label].append(idx)
    arr = []
    for v in nodes.values():
        arr += list(permutations(v, 2))
    return torch.tensor(arr, dtype=torch.long).T, torch.tensor(mask, dtype=torch.long)