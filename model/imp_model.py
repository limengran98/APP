import torch
from torch import nn

from model.modules import ProtoAwarePropagation, EfficientPseudoLabel

class HeteroImputation(nn.Module):
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