"""
Helper functions for training and evaluation.
"""
import torch


def RMSE(data1, data2):
    """Root Mean Square Error."""
    return torch.sqrt(torch.mean((data1 - data2)**2)).item()


def MAE(data1, data2):
    """Mean Absolute Error."""
    return torch.mean(torch.abs(data1 - data2)).item()


def update_learning_rate(optimizer, epoch, initial_lr=5e-5):
    """
    Update learning rate based on training epoch.
    
    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        initial_lr: Initial learning rate
    
    Returns:
        float: Updated learning rate
    """
    if epoch < 50:
        new_lr = initial_lr * 10
    elif epoch < 80:
        new_lr = initial_lr * 9.5
    elif epoch < 160:
        new_lr = initial_lr * 5.5
    elif epoch < 300:
        new_lr = initial_lr * 1.0
    elif epoch < 350:
        new_lr = initial_lr * 0.05
    elif epoch < 800:
        new_lr = initial_lr * 0.001
    elif epoch < 1001:
        new_lr = initial_lr * 0.0005
    # elif epoch < 1400:
    #     new_lr = initial_lr * 0.005
    # elif epoch < 1800:
    #     new_lr = initial_lr * 0.001
    # elif epoch < 2000:
    #     new_lr = initial_lr * 0.0005
    # elif epoch < 2500:
    #     new_lr = initial_lr * 0.0001
    # else:
    #     new_lr = initial_lr * 0.00001

    # elif epoch < 1400:
    #     new_lr = initial_lr * 0.0005
    # elif epoch < 1800:
    #     new_lr = initial_lr * 0.0001
    # elif epoch < 2000:
    #     new_lr = initial_lr * 0.00005
    # elif epoch < 2500:
    #     new_lr = initial_lr * 0.00001
    # else:
    #     new_lr = initial_lr * 0.000001    

    elif epoch < 1400:
        new_lr = initial_lr * 0.05
    elif epoch < 1800:
        new_lr = initial_lr * 0.01
    elif epoch < 2000:
        new_lr = initial_lr * 0.005
    elif epoch < 2500:
        new_lr = initial_lr * 0.001
    else:
        new_lr = initial_lr * 0.0001   
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    # print(f"Learning rate updated to: {new_lr}")
    return new_lr
