"""
Online learning dataset loader for ACOPF with distribution shift scenarios.

Data Structure:
- Labeled Training Set: 1000 samples (X_con_1354_train.txt, X_in_1354_train.txt, X_other_information_1354_train.txt)
- Unlabeled Fine-tuning Set: 2000 samples (first 2000 of test set, only X_con - state/input features)
- Labeled Test Set: 500 samples (last 500 of test set, full labels for evaluation)

Scenarios:
1. Load Variation: Different load distributions (-0.1, -0.05, 0.0, 0.05, 0.1 mean shifts)
2. Topology Change: Line outages (lines 9, 7, 20, 19, 17)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os


def load_labeled_data(data_dir, filename_prefix):
    """
    Load labeled data (X_con, X_in, X_other).
    
    Args:
        data_dir: Directory containing data files
        filename_prefix: Prefix for filenames (e.g., "1354_train" or "1354_test_0.0")
    
    Returns:
        tuple: (X_con, X_in, X_other) as numpy arrays
    """
    X_con = []
    with open(f"{data_dir}/X_con_{filename_prefix}.txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            X_con.append([float(x) for x in data])
    
    X_in = []
    with open(f"{data_dir}/X_in_{filename_prefix}.txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            X_in.append([float(x) for x in data])
    
    X_other = []
    with open(f"{data_dir}/X_other_information_{filename_prefix}.txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            X_other.append([float(x) for x in data])

    X_con = np.array(X_con) / 100.0
    
    X_in = np.array(X_in)
    X_in[:, :260] = X_in[:, :260] / 100.0
    
    return X_con, X_in, np.array(X_other)


def load_unlabeled_data(data_dir, filename_prefix):
    """
    Load unlabeled data (X_con only) for fine-tuning.
    
    Args:
        data_dir: Directory containing data files
        filename_prefix: Prefix for filenames
    
    Returns:
        X_con as numpy array
    """
    X_con = []
    with open(f"{data_dir}/X_con_{filename_prefix}.txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            X_con.append([float(x) for x in data])
    
    return np.array(X_con)


def prepare_online_learning_data(data_dir="./CompleteDataSet",
                                  load_variation=True,
                                  topology_change=True,
                                  load_scenarios=['-0.03', '-0.01', '0.0', '0.01', '0.03']):
    """
    Prepare data for online learning experiments.
    
    Args:
        data_dir: Directory containing data files
        load_variation: Whether to load load variation scenarios
        topology_change: Whether to load topology change scenarios
    
    Returns:
        dict: Dictionary containing all data splits
    """
    data = {
        'labeled_train': None,
        'load_variation': {},
        'topology_change': {}
    }
    
    # 1. Load labeled training set (10000 samples)
    print("Loading labeled training set...")
    X_con_train, X_in_train, X_other_train = load_labeled_data(data_dir, "1354_train")
    data['labeled_train'] = {
        'X_con': X_con_train,
        'X_in': X_in_train,
        'X_other': X_other_train
    }
    print(f"  Labeled train: {data['labeled_train']['X_con'].shape}")
    
    # 2. Load variation scenarios (only if needed)
    if load_variation:
        # load_dis_scenarios = ['0.0']
        for scenario in load_scenarios:
            print(f"Loading load variation scenario: {scenario}")
            X_con, X_in, X_other = load_labeled_data(data_dir, f"1354_test_{scenario}")
            
            # First 2000 for fine-tuning (labels kept for optional semi-supervised use)
            # Last 500 for evaluation
            data['load_variation'][scenario] = {
                'finetune_data': {
                    'X_con':   X_con[:2000],
                    'X_in':    X_in[:2000],
                    'X_other': X_other[:2000],
                },
                'test_labeled': {
                    'X_con':   X_con[2000:],
                    'X_in':    X_in[2000:],
                    'X_other': X_other[2000:]
                }
            }
            print(f"  Finetune data: {data['load_variation'][scenario]['finetune_data']['X_con'].shape}")
            print(f"  Test (labeled): {data['load_variation'][scenario]['test_labeled']['X_con'].shape}")
    else:
        print("Skipping load variation data (--skip_load_variation)")
    
    # 3. Topology change scenarios (only if needed)
    if topology_change:
        topo_scenarios = ['17', '19', '20', '37', '53']
        for scenario in topo_scenarios:
            print(f"Loading topology change scenario: line {scenario}")
            X_con, X_in, X_other = load_labeled_data(data_dir, f"1354_test_TopologyChange_{scenario}")
            
            # First 2000 for fine-tuning (labels kept for optional semi-supervised use)
            # Last 500 for evaluation
            data['topology_change'][scenario] = {
                'finetune_data': {
                    'X_con':   X_con[:2000],
                    'X_in':    X_in[:2000],
                    'X_other': X_other[:2000],
                },
                'test_labeled': {
                    'X_con':   X_con[2000:],
                    'X_in':    X_in[2000:],
                    'X_other': X_other[2000:]
                }
            }
            print(f"  Finetune data: {data['topology_change'][scenario]['finetune_data']['X_con'].shape}")
            print(f"  Test (labeled): {data['topology_change'][scenario]['test_labeled']['X_con'].shape}")
    else:
        print("Skipping topology change data (--skip_topology_variation)")
    
    return data

def create_tensors(X_con, X_in=None, X_other=None, device='cpu'):
    """
    Create PyTorch tensors from numpy arrays.
    
    Args:
        X_con: State data
        X_in: Action data (optional)
        X_other: Other information (optional)
        device: Device to place tensors on
    
    Returns:
        PyTorch tensors
    """
    X_con_tensor = torch.tensor(X_con, dtype=torch.float32, device=device)
    
    tensors = [X_con_tensor]
    
    if X_in is not None:
        X_in_tensor = torch.tensor(X_in, dtype=torch.float32, device=device)
        tensors.append(X_in_tensor)
    
    if X_other is not None:
        X_other_tensor = torch.tensor(X_other, dtype=torch.float32, device=device)
        tensors.append(X_other_tensor)
    
    return tuple(tensors)

def create_dataloaders(X_con, X_in=None, X_other=None, batch_size=64, shuffle=True):
    """
    Create PyTorch DataLoaders.
    
    Args:
        X_con: State data
        X_in: Action data (optional)
        X_other: Other information (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader
    """
    X_con_tensor = torch.tensor(X_con, dtype=torch.float32)
    
    if X_in is not None and X_other is not None:
        # Labeled data
        X_in_tensor = torch.tensor(X_in, dtype=torch.float32)
        X_other_tensor = torch.tensor(X_other, dtype=torch.float32)
        dataset = TensorDataset(X_con_tensor, X_in_tensor, X_other_tensor)
    else:
        # Unlabeled data (only states)
        dataset = TensorDataset(X_con_tensor)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
