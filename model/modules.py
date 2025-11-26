import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

class ProtoAwarePropagation(nn.Module):
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
    def __init__(self, input_dim, num_prototypes):
        super().__init__()
        self.prototype_learner = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_prototypes)
        )
        self.topk = 20  
        
    def forward(self, x, edge_index):
        row, col = edge_index
        # Safe sampling for large graphs
        if len(row) > 100000:
             sampled = torch.randint(0, len(row), (len(row)//5,))
             row_s, col_s = row[sampled], col[sampled]
        else:
             row_s, col_s = row, col
             
        proto_logits = self.prototype_learner(x)
        smooth_proto = scatter_mean(proto_logits[col_s], row_s, dim=0, dim_size=x.size(0))
        pseudo_labels = smooth_proto.argmax(dim=1)
        confidence = F.softmax(proto_logits, dim=1).max(dim=1)[0]
        return pseudo_labels, confidence

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
    def forward(self, x_filled, x_PVP, a, b):
        fused = a * x_filled + b * x_PVP
        fused += 0.01 * torch.tanh(x_filled * x_PVP)
        return fused