import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from scipy import io
import torch_geometric.utils
from torch_geometric.utils import to_undirected, is_undirected

from data_load.dataset_def import HeterophilousGraphDataset

from torch_geometric.datasets import LINKXDataset
try:
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:
    PygNodePropPredDataset = None


def is_large(data):
    return data == 'arxiv' or data == 'Genius'


def is_continuous(data):
    return data in ['pubmed', 'coauthor', 'arxiv']


def validate_edges(edges):
    # No self-loops
    for src, dst in edges.t():
        if src.item() == dst.item():
            raise ValueError()
    # Each edge (a, b) appears only once.
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if dst in m[src]:
            raise ValueError()
        m[src].add(dst)
    # Each pair (a, b) and (b, a) exists together.
    for src, neighbors in m.items():
        for dst in neighbors:
            if src not in m[dst]:
                raise ValueError()


def load_data(args, split=None, seed=None, verbose=False, normalize=False,
              validate=False):
    """
    Load a dataset from its name.
    """
    root = './data'
    dataset = args.data
    seed = 0

    # 1. Original 5 Heterophilous Datasets
    if dataset in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers"]:
        data = HeterophilousGraphDataset(root, name=dataset)
        # Check if we need to unpack list (PyG datasets usually return list-like)
        if isinstance(data, HeterophilousGraphDataset):
            data = data[0]

    # 2. Squirrel/Chameleon legacy handling
    elif dataset in ['squirrel', 'chameleon']:
        # Path to data files
        DATAPATH = "./data/"
        fulldata = io.loadmat(f'{DATAPATH}/{dataset}.mat')
        # Load data from .mat file
        edge_index = fulldata['edge_index']      
        node_feat = fulldata['node_feat']        
        label = np.array(fulldata['label'], dtype=np.int32).flatten() 
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(node_feat, dtype=torch.float)   
        y = torch.tensor(label, dtype=torch.long)        
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
        data = Data(x=x, edge_index=edge_index, y=y)

    # --- New: 2. LINKXDataset (Genius) ---
    elif dataset == 'Genius':
        print(f"Loading Genius from LINKXDataset...")
        try:
            dataset_obj = LINKXDataset(root=root, name='genius')
            data = dataset_obj[0]
        except Exception as e:
            raise ValueError(f"Failed to load Genius: {e}")

        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)

    # --- New: 3. arXiv-year (requires OGB) ---
    elif dataset == 'arxiv-year':
        if PygNodePropPredDataset is None:
            raise ImportError("Please install ogb: 'pip install ogb'")
        
        print("Loading ogbn-arxiv for year prediction...")
        try:
            dataset_obj = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
            data = dataset_obj[0]
        except Exception as e:
             raise ValueError(f"Failed to load ogbn-arxiv: {e}")
        
        # --- FIX: Binning years into 5 classes (Standard LinkX Setting) ---
        if hasattr(data, 'node_year'):
            y = data.node_year.squeeze().float()
            
            # Use Quantile Binning to get 5 balanced classes
            print("Binning arXiv years into 5 classes (LinkX standard)...")
            quantiles = torch.tensor([0.2, 0.4, 0.6, 0.8])
            boundaries = torch.quantile(y, quantiles)
            y_binned = torch.bucketize(y, boundaries)
            
            data.y = y_binned.long()
            print(f"Classes created: {data.y.unique().tolist()}")
        else:
            raise ValueError("Attribute 'node_year' not found in ogbn-arxiv.")
            
        # Ensure undirected for consistency
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    else:
        raise ValueError(dataset)
    
    # Unified extraction of x, y, edges variables
    if hasattr(data, 'data'): # Handle weird wrapper cases if any
        node_x = data.data.x
        node_y = data.data.y
        edges = data.data.edge_index
    else:
        node_x = data.x
        node_y = data.y
        edges = data.edge_index

    if validate:
        validate_edges(edges)

    if normalize:
        assert (node_x < 0).sum() == 0  
        norm_x = node_x.clone()
        norm_x[norm_x.sum(dim=1) == 0] = 1
        norm_x = norm_x / norm_x.sum(dim=1, keepdim=True)
        node_x = norm_x

    # Split Logic
    if split is None:
        if hasattr(data, 'train_mask'):
            # Handle cases where mask might be multi-column (e.g., Roman-empire has 10 splits)
            if data.train_mask.dim() > 1:
                trn_mask = data.train_mask[:, seed % data.train_mask.shape[1]]
                val_mask = data.val_mask[:, seed % data.val_mask.shape[1]]
            else:
                trn_mask = data.train_mask
                val_mask = data.val_mask
            
            trn_nodes = torch.nonzero(trn_mask).view(-1)
            # val_nodes = torch.nonzero(val_mask).view(-1) # Unused but logic preserved
            # test_nodes calculation relies on what's left or explicit mask
            if hasattr(data, 'test_mask'):
                 if data.test_mask.dim() > 1:
                     test_mask = data.test_mask[:, seed % data.test_mask.shape[1]]
                 else:
                     test_mask = data.test_mask
                 test_nodes = torch.nonzero(test_mask).view(-1)
            else:
                 test_nodes = torch.nonzero(~(trn_mask | val_mask)).view(-1)
        else:
            trn_nodes, val_nodes, test_nodes = None, None, None

    elif len(split) == 2:
        # Manual random split (for Genius/Arxiv if they lack masks or user overrides)
        trn_size, test_size = split
        indices = np.arange(node_x.shape[0])
        # Use stratify to ensure class balance
        try:
            trn_nodes, test_nodes = train_test_split(indices, test_size=test_size, random_state=seed,
                                                     stratify=node_y.cpu().numpy())
        except:
            # Fallback if stratify fails (e.g. class with 1 sample)
            trn_nodes, test_nodes = train_test_split(indices, test_size=test_size, random_state=seed)

        trn_nodes = torch.from_numpy(trn_nodes)
        test_nodes = torch.from_numpy(test_nodes)
    else:
       raise ValueError(split)

    # Unified return values
    return data, trn_nodes, test_nodes