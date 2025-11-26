import warnings
warnings.filterwarnings("ignore")


import argparse
import os
import time
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
from torch_geometric.utils import subgraph

from data_load.data_loader import load_data
from utils.graph_ops import to_device
from utils.propagations import APA
from utils.baselines import MLP, GCN, GAT, AP
from model.propagation_layer import arbLabel
from model.imp_model import HeteroImputation
from model.modules import AdaptiveFeatureFusion
from model.clustering import robust_pseudo_labels, save_or_load_x_PVP
from model.loss_funcs import contrastive_proto_loss

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Roman-empire') # ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers",'squirrel', 'Genius', 'arxiv-year']
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--missrate', type=float, default=0.6) 
    parser.add_argument('--alpha', type=float, default=1) # 1
    parser.add_argument('--beta', type=float, default=0) # 0
    parser.add_argument('--gamma', type=float, default=0.9) # 0.9
    parser.add_argument('--num_iter', type=int, default=2)
    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.01)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-2) 
    parser.add_argument('--c_lr', type=float, default=1e-3) 
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=300)
    parser.add_argument('--model', type=str, default='MLP') 
    parser.add_argument('--save_results', action='store_true', help='Save results to CSV')

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    device = to_device(args.gpu)
    data, trn_nodes, test_nodes = load_data(args, split=(1-args.missrate, args.missrate), seed=args.seed)
    y_all = data.y

    x_all = data.x.clone()
    x_missing = x_all.clone()
    x_missing[test_nodes] = 0
    num_nodes = x_all.size(0)
    num_classes = (data.y.max() + 1).item()

    # Initial propagation
    propagation = APA(data.edge_index, x_all, trn_nodes)
    x_missing_filled = propagation.fp(x_missing)
    x_all = x_all.to(device)
    edge_index = data.edge_index.to(device)
    
    

    model = HeteroImputation(
        input_dim=data.x.size(1), 
        num_prototypes=num_classes
    ).to(device)



    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20) 

    prev_loss = float('inf')
    prev_prev_loss = float('inf')

    for epoch in range(200):
        prototype_matrix, pseudo_labels = model(x_missing_filled.to(device), edge_index)
        rec_loss = F.mse_loss(prototype_matrix[trn_nodes], x_all[trn_nodes])
        proto_loss = contrastive_proto_loss(prototype_matrix, model.proto_prop.prototypes)
        total_loss = rec_loss + args.mu*proto_loss
        prototype_matrix = prototype_matrix.detach()
        #print(f"Epoch [{epoch+1}] | Total Loss: {total_loss:.4f}")
        if total_loss > prev_loss and prev_loss > prev_prev_loss:
            #print(f"Loss increased in the last two epochs. Stopping early.")
            break
        prev_prev_loss = prev_loss
        prev_loss = total_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step() 

    best_acc, best_nmi, best_ari, best_f1, pseudo_labels = robust_pseudo_labels(x_missing, y_all, data.edge_index, num_classes)
    pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)
    propagation_model = arbLabel(data.edge_index, x_all.cpu(), pseudo_labels, trn_nodes)
    x_PVP = save_or_load_x_PVP(x_missing, propagation_model, args, device)

    fusion = AdaptiveFeatureFusion(input_dim=prototype_matrix.size(1))

    if args.data in ["Roman-empire", "Minesweeper"]:
        x_feature = fusion(prototype_matrix, x_PVP, a=0.001, b=1)
    else:
        x_feature = fusion(prototype_matrix, x_PVP, a=1, b=0.001)
    print(x_feature.shape)

    x_all = x_all.to(device)
    x_feature = x_feature.to(device)
    

    # Reconstruction metrics
    mse = mean_squared_error(x_feature[test_nodes].cpu(), x_all[test_nodes].cpu())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(x_feature[test_nodes].cpu(), x_all[test_nodes].cpu())
    Restructure = [mse, rmse, mae]
    print(f'MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}')

    # Classification setup
    test_accuracies, test_f1s, test_roc_aucs = [], [], []
    val_accuracies, val_f1s, val_roc_aucs = [], [], []

    edge_index, _ = subgraph(test_nodes.to(device), edge_index, relabel_nodes=True)
    edge_index = edge_index.to(device)

    for run in range(10):
        seed = 0 + run  
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if args.data == 'squirrel':
            x_feature = torch.Tensor(x_feature)
            
        X = x_feature[test_nodes].to(device)
        labels = data.y[test_nodes].to(device)


        train_ratio = 0.5
        val_ratio = 0.25
        test_ratio = 0.25
        num_nodes = X.shape[0]
        indices = list(range(num_nodes))
        

        train_indices, temp_indices = train_test_split(indices, train_size=train_ratio, random_state=seed)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)

        train_indices = torch.tensor(train_indices, device=device)
        val_indices = torch.tensor(val_indices, device=device)
        test_indices = torch.tensor(test_indices, device=device)


        input_dim = X.shape[1]
        hidden_dim = args.hidden_size
        output_dim = len(torch.unique(labels))

        if args.model == 'MLP':
            model = MLP(input_dim, hidden_dim, output_dim).to(device)
            #model = GraphAwareMLP(input_dim, hidden_dim, output_dim).to(device)
        elif args.model == 'GCN':
            model = GCN(input_dim, hidden_dim, output_dim).to(device)
        elif args.model == 'GAT':
            model = GAT(input_dim, hidden_dim, output_dim, heads=args.heads).to(device)
        elif args.model == 'AP':
            model = AP(input_dim, hidden_dim, output_dim, args).to(device)
        else:
            raise ValueError("Unsupported model type. Choose from 'MLP', 'GCN', or 'GAT'.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.c_lr)



        # Early stopping parameters
        patience = args.patience
        epochs_without_improvement = 0

        num_epochs = args.epochs
        best_val_acc = 0
        best_val_f1 = 0
        best_val_roc_auc = 0
        best_test_acc = 0
        best_test_f1 = 0
        best_test_roc_auc = 0
        best_model_weights = None

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            if args.model == 'MLP': 
                out, re_x = model(X[train_indices])
                OUT, _ = model(X)
                _, train_predicted = torch.max(out, 1)
                criterion_loss = criterion(out, labels[train_indices])
                train_loss = criterion_loss 
            else:  
                out = model(X, edge_index)
                train_loss = criterion(out[train_indices], labels[train_indices])
                _, train_predicted = torch.max(out, 1)
                train_predicted = train_predicted[train_indices]
            train_loss.backward()
            optimizer.step()

            
            train_acc = accuracy_score(labels[train_indices].cpu(), train_predicted.cpu())
            train_f1 = f1_score(labels[train_indices].cpu(), train_predicted.cpu(), average='weighted')

            model.eval()
            with torch.no_grad():
                if args.model == 'MLP':
                    val_out, _ = model(X[val_indices])
                else:
                    val_out = model(X, edge_index)
                    val_out = val_out[val_indices]
                _, val_predicted = torch.max(val_out, 1)
                val_acc = accuracy_score(labels[val_indices].cpu(), val_predicted.cpu())
                val_f1 = f1_score(labels[val_indices].cpu(), val_predicted.cpu(), average='weighted')
                val_prob = torch.softmax(val_out, dim=1).cpu().numpy()

                if args.data == "Minesweeper" or args.data == "Tolokers" or args.data == "Genius":
                    val_roc_auc = roc_auc_score(labels[val_indices].cpu().numpy(), val_prob[:, 1])
                else:
                    val_roc_auc = roc_auc_score(labels[val_indices].cpu().numpy(), val_prob, average='weighted', multi_class='ovr')

                if args.model == 'MLP':
                    test_out, _  = model(X[test_indices])
                    _, test_predicted = torch.max(test_out, 1)
                else:
                    test_out = model(X, edge_index)
                    test_out = test_out[test_indices]
                _, test_predicted = torch.max(test_out, 1)
                test_acc = accuracy_score(labels[test_indices].cpu(), test_predicted.cpu())
                test_f1 = f1_score(labels[test_indices].cpu(), test_predicted.cpu(), average='weighted')
                test_prob = torch.softmax(test_out, dim=1).cpu().numpy()

                if args.data == "Minesweeper" or args.data == "Tolokers" or args.data == "Genius":
                    test_roc_auc = roc_auc_score(labels[test_indices].cpu().numpy(), test_prob[:, 1])
                else:
                    test_roc_auc = roc_auc_score(labels[test_indices].cpu().numpy(), test_prob, average='weighted', multi_class='ovr')

            if val_roc_auc > best_val_roc_auc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_roc_auc = val_roc_auc
                best_test_acc = test_acc
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement += 1
                # Early stopping
            # if epochs_without_improvement >= patience:
            #     print("Early stopping at epoch", epoch)
            #     break

        model.load_state_dict(best_model_weights)
        model.eval()

        with torch.no_grad():
            if args.model == 'MLP':
                final_val_out, _  = model(X[val_indices])
            else:
                final_val_out = model(X, edge_index)
                final_val_out = final_val_out[val_indices]
            _, final_val_predicted = torch.max(final_val_out, 1)
            final_val_acc = accuracy_score(labels[val_indices].cpu(), final_val_predicted.cpu())
            final_val_f1 = f1_score(labels[val_indices].cpu(), final_val_predicted.cpu(), average='weighted')

            if args.model == 'MLP':
                final_test_out, _  = model(X[test_indices])
            else:
                final_test_out = model(X, edge_index)
                final_test_out = final_test_out[test_indices]
            _, final_test_predicted = torch.max(final_test_out, 1)
            final_test_acc = accuracy_score(labels[test_indices].cpu(), final_test_predicted.cpu())
            final_test_f1 = f1_score(labels[test_indices].cpu(), final_test_predicted.cpu(), average='weighted')

            # ROC AUC
            final_val_prob = torch.softmax(final_val_out, dim=1).cpu().numpy()
            final_test_prob = torch.softmax(final_test_out, dim=1).cpu().numpy()

            if args.data == "Minesweeper" or args.data == "Tolokers" or args.data == "Genius":
                final_val_roc_auc = roc_auc_score(labels[val_indices].cpu().numpy(), final_val_prob[:, 1])
                final_test_roc_auc = roc_auc_score(labels[test_indices].cpu().numpy(), final_test_prob[:, 1])
            else:
                final_val_roc_auc = roc_auc_score(labels[val_indices].cpu().numpy(), final_val_prob, average='weighted', multi_class='ovr')
                final_test_roc_auc = roc_auc_score(labels[test_indices].cpu().numpy(), final_test_prob, average='weighted', multi_class='ovr')
                
            val_accuracies.append(final_val_acc)
            val_f1s.append(final_val_f1)
            val_roc_aucs.append(final_val_roc_auc)

            test_accuracies.append(final_test_acc)
            test_f1s.append(final_test_f1)
            test_roc_aucs.append(final_test_roc_auc)


            print(f'Run {run+1} - Final Validation Accuracy: {final_val_acc*100:.2f}, Final Validation F1: {final_val_f1*100:.2f}, Final Validation ROC AUC: {final_val_roc_auc*100:.2f}')
            print(f'Run {run+1} - Final Test Accuracy: {final_test_acc*100:.2f}, Final Test F1: {final_test_f1*100:.2f}, Final Test ROC AUC: {final_test_roc_auc*100:.2f}')

    mean_val_acc = np.mean(val_accuracies)
    std_val_acc = np.std(val_accuracies)
    mean_val_f1 = np.mean(val_f1s)
    std_val_f1 = np.std(val_f1s)
    mean_val_roc_auc = np.mean(val_roc_aucs)
    std_val_roc_auc = np.std(val_roc_aucs)

    mean_test_acc = np.mean(test_accuracies)
    std_test_acc = np.std(test_accuracies)
    mean_test_f1 = np.mean(test_f1s)
    std_test_f1 = np.std(test_f1s)
    mean_test_roc_auc = np.mean(test_roc_aucs)
    std_test_roc_auc = np.std(test_roc_aucs)


    print(f'\nAverage Validation Accuracy: {mean_val_acc*100:.2f} ± {std_val_acc*100:.2f}')
    print(f'Average Validation F1: {mean_val_f1*100:.2f} ± {std_val_f1*100:.2f}')
    print(f'Average Validation ROC AUC: {mean_val_roc_auc*100:.2f} ± {std_val_roc_auc*100:.2f}')
    print(f'Average Test Accuracy: {mean_test_acc*100:.2f} ± {std_test_acc*100:.2f}')
    print(f'Average Test F1: {mean_test_f1*100:.2f} ± {std_test_f1*100:.2f}')
    print(f'Average Test ROC AUC: {mean_test_roc_auc*100:.2f} ± {std_test_roc_auc*100:.2f}')

    # torch.save(OUT, os.path.join(output_dir, 'oout.pt'))



    if args.save_results:
        result = [
            Restructure[0],  # MSE
            Restructure[1],  # RMSE
            Restructure[2],  # MAE
            np.mean(test_accuracies),  # acc
            np.mean(test_f1s),         # f1
            np.mean(test_roc_aucs)     # auc
        ]
        os.makedirs('./result', exist_ok=True)
        with open(f'./result/{args.data}results.csv', 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow(result)
        print(f"Results saved to ./result/{args.data}results.csv")

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    elapsed = end - start
    print(f"Total runtime: {elapsed:.2f} seconds")