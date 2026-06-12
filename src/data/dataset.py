"""
Dataset loading and preprocessing for ACOPF training.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(data_dir="./Dataset"):
    """
    Load training and testing data from files.
    
    Args:
        data_dir: Directory containing data files
    
    Returns:
        tuple: (X_con_train, X_in_train, X_other_train, 
                X_con_test, X_in_test, X_other_test)
    """
    # Load training data
    X_con_train = []
    with open(f"{data_dir}/X_con_1354_train.txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            X_con_train.append([float(x) for x in data])

    X_in_train = []
    with open(f"{data_dir}/X_in_1354_train.txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            X_in_train.append([float(x) for x in data])

    X_other_train = []
    with open(f"{data_dir}/X_other_information_1354_train.txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            X_other_train.append([float(x) for x in data])

    # Load test data
    X_con_test = []
    with open(f"{data_dir}/X_con_1354_test_0.05.txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            X_con_test.append([float(x) for x in data])

    X_in_test = []
    with open(f"{data_dir}/X_in_1354_test_0.05.txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            X_in_test.append([float(x) for x in data])

    X_other_test = []
    with open(f"{data_dir}/X_other_information_1354_test_0.05.txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            X_other_test.append([float(x) for x in data])

    # return (
    #     np.array(X_con_train)[:9900,:],
    #     np.array(X_in_train)[:9900,:],
    #     np.array(X_other_train)[:9900,:],
    #     np.array(X_con_train)[-100:,:],
    #     np.array(X_in_train)[-100:,:],
    #     np.array(X_other_train)[-100:,:]
    # )
    return (
        np.array(X_con_train),
        np.array(X_in_train),
        np.array(X_other_train),
        np.array(X_con_train)[-100:,:],
        np.array(X_in_train)[-100:,:],
        np.array(X_other_train)[-100:,:]
    )

def prepare_tensors(X_con_train, X_in_train, X_other_train,
                   X_con_test, X_in_test, X_other_test):
    """
    Prepare and normalize data tensors.
    
    Args:
        X_con_train: Training state data
        X_in_train: Training action data
        X_other_train: Training auxiliary data
        X_con_test: Test state data
        X_in_test: Test action data
        X_other_test: Test auxiliary data
    
    Returns:
        tuple: PyTorch tensors for training and testing
    """
    # Normalize data
    X_in_train_norm = X_in_train.copy()
    X_in_train_norm[:, :260] = X_in_train_norm[:, :260] / 100
    
    X_con_train_norm = X_con_train / 100
    
    X_in_test_norm = X_in_test.copy()
    X_in_test_norm[:, :260] = X_in_test_norm[:, :260] / 100
    
    X_con_test_norm = X_con_test / 100

    # Convert to PyTorch tensors
    X_in_train_tensor = torch.tensor(X_in_train_norm, dtype=torch.float32)
    X_con_train_tensor = torch.tensor(X_con_train_norm, dtype=torch.float32)
    X_other_train_tensor = torch.tensor(X_other_train, dtype=torch.float32)

    X_in_test_tensor = torch.tensor(X_in_test_norm, dtype=torch.float32)
    X_con_test_tensor = torch.tensor(X_con_test_norm, dtype=torch.float32)
    X_other_test_tensor = torch.tensor(X_other_test, dtype=torch.float32)

    print(f"Training data shape: X_in={X_in_train.shape}, X_con={X_con_train.shape}")
    print(f"Test data shape: X_in={X_in_test.shape}, X_con={X_con_test.shape}")

    return (
        X_in_train_tensor, X_con_train_tensor, X_other_train_tensor,
        X_in_test_tensor, X_con_test_tensor, X_other_test_tensor
    )


def create_data_loaders(X_in_train, X_con_train, X_other_train,
                       X_in_test, X_con_test, X_other_test,
                       batch_size=64, shuffle_train=True):
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        X_in_train: Training input tensor
        X_con_train: Training state tensor
        X_other_train: Training auxiliary tensor
        X_in_test: Test input tensor
        X_con_test: Test state tensor
        X_other_test: Test auxiliary tensor
        batch_size: Batch size for training
        shuffle_train: Whether to shuffle training data
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_dataset = TensorDataset(X_in_train, X_con_train, X_other_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=shuffle_train)

    test_dataset = TensorDataset(X_in_test, X_con_test, X_other_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False)

    return train_loader, test_loader
