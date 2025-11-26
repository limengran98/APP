import os
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score,
    f1_score
)

from model.graph_tools import LARGE_GRAPH_THRESHOLD

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
        x_PVP = torch.load(filepath, map_location=device)  
    else:
        print(f"File {filename} not found. Computing and saving.")
        x_PVP = propagation_model.arb_label_100(
            x_missing, alpha=args.alpha, beta=args.beta, gamma=args.gamma, num_iter=args.num_iter
        ).to(device)
        torch.save(x_PVP, filepath)
    return x_PVP.to(device) 


def robust_pseudo_labels(x, y_all, edge_index, num_classes):
    """Robust pseudo-label generation with scalable clustering"""
    
    # --- Optimization: Select clustering method based on data scale ---
    if x.size(0) < LARGE_GRAPH_THRESHOLD:
        cluster_models = [
            KMeans(n_clusters=num_classes),
            MiniBatchKMeans(n_clusters=num_classes),
            AgglomerativeClustering(n_clusters=num_classes),
        ]
    else:
        # Use MiniBatch for large datasets, run multiple times for stability
        cluster_models = [
            MiniBatchKMeans(n_clusters=num_classes, batch_size=2048, n_init=3),
            MiniBatchKMeans(n_clusters=num_classes, batch_size=4096, n_init=3),
            MiniBatchKMeans(n_clusters=num_classes, batch_size=2048, n_init=3),
            MiniBatchKMeans(n_clusters=num_classes, batch_size=4096, n_init=3),
            MiniBatchKMeans(n_clusters=num_classes, batch_size=2048, n_init=3),
            MiniBatchKMeans(n_clusters=num_classes, batch_size=4096, n_init=3),
        ]

    all_preds = []
    x_np = x.cpu().numpy()
    
    for model in cluster_models:
        try:
            pred = model.fit_predict(x_np)
            all_preds.append(torch.tensor(pred))
        except:
            continue
    
    # Guard against empty predictions
    if not all_preds:
        return 0,0,0,0, torch.zeros(x.size(0))

    consensus_labels = torch.mode(torch.stack(all_preds), dim=0).values
    return evaluate_and_return(consensus_labels.cpu().numpy(), y_all.cpu().numpy())


def evaluate_and_return(labels, y_true):
    nmi = normalized_mutual_info_score(y_true, labels)
    ac = accuracy_score(y_true, labels)
    f1 = f1_score(y_true, labels, average='micro')
    ari = adjusted_rand_score(y_true, labels)
    print(f'AC: {ac:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}')
    return ac, nmi, ari, f1, labels