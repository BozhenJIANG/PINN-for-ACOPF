"""
Training step functions for ACOPF model.
"""
import torch
from .loss import compute_loss

def train_step_0(model, x, y, other_variable, optimizer, epoch, 
                pre_train_epoch, penalty_coefficient, case1354):
    """
    Initial training step - trains both encoder and PINN_PF.
    
    This is used during pre-training to align encoder and PINN_PF.
    
    Args:
        model: ACOPF model
        x: Ground truth actions
        y: States
        other_variable: Auxiliary variables
        optimizer: Optimizer
        epoch: Current epoch
        pre_train_epoch: Number of pre-training epochs
        penalty_coefficient: Penalty coefficient
        case1354: Power system case data
    
    Returns:
        tuple: Loss values
    """
    optimizer.zero_grad()
    losses = compute_loss(model, x, y, other_variable, epoch, 
                         pre_train_epoch, penalty_coefficient, case1354)
    
    # Combine encoder loss + PINN_PF loss in a single backward pass
    # (two separate backward calls with optimizer.step() in between would
    #  invalidate the retained graph due to in-place parameter updates)
    # print(losses)
    loss = losses[0] + losses[1]
    loss.backward()
    optimizer.step()
    
    return losses


def train_step_1_1(model, x, y, other_variable, optimizer, epoch, 
                  pre_train_epoch, penalty_coefficient, case1354,
                  w_p_balance=1.0, w_q_balance=1.0, w_theta_balance=1.0):
    """
    Training step 1.1 - trains PINN_PF model only.
    
    Focuses on power balance and physics constraints.
    
    Args:
        model: ACOPF model
        x: Ground truth actions
        y: States
        other_variable: Auxiliary variables
        optimizer: Optimizer
        epoch: Current epoch
        pre_train_epoch: Number of pre-training epochs
        penalty_coefficient: Penalty coefficient
        case1354: Power system case data
        w_p_balance: Weight for P_balance loss (losses[1])
        w_q_balance: Weight for Q_balance loss (losses[2])
        w_theta_balance: Weight for theta_balance loss (losses[3])
    
    Returns:
        tuple: Loss values
    """
    optimizer.zero_grad()
    losses = compute_loss(model, x, y, other_variable, epoch, 
                         pre_train_epoch, penalty_coefficient, case1354)
    
    # Only train PINN_PF, not encoder
    # Focus on: P_balance, Q_balance, theta_balance (with configurable weights)
    loss = (0 * losses[0] + w_p_balance * losses[1] + w_q_balance * losses[2] + 
            w_theta_balance * losses[3] + 
            0 * losses[4] + 0 * losses[5] + 0 * losses[6] + 0 * losses[7] + 
            0 * losses[8] + 0 * losses[9] + 0 * losses[10] + 0 * losses[11])
    
    loss.backward()
    optimizer.step()
    return losses


def train_step_1_2(model, x, y, other_variable, optimizer, epoch, 
                  pre_train_epoch, penalty_coefficient, case1354,
                  w_p_balance=1.0, w_q_balance=1.0, w_theta_balance=1.0):
    """
    Training step 1.2 - trains both PINN_PF and encoder.
    
    Similar to 1_1 but can be used for joint training.
    
    Args:
        model: ACOPF model
        x: Ground truth actions
        y: States
        other_variable: Auxiliary variables
        optimizer: Optimizer
        epoch: Current epoch
        pre_train_epoch: Number of pre-training epochs
        penalty_coefficient: Penalty coefficient
        case1354: Power system case data
        w_p_balance: Weight for P_balance loss (losses[1])
        w_q_balance: Weight for Q_balance loss (losses[2])
        w_theta_balance: Weight for theta_balance loss (losses[3])
    
    Returns:
        tuple: Loss values
    """
    optimizer.zero_grad()
    losses = compute_loss(model, x, y, other_variable, epoch, 
                         pre_train_epoch, penalty_coefficient, case1354)
    
    loss = (0 * losses[0] + w_p_balance * losses[1] + w_q_balance * losses[2] + 
            w_theta_balance * losses[3] + 
            0 * losses[4] + 0 * losses[5] + 0 * losses[6] + 0 * losses[7] + 
            0 * losses[8] + 0 * losses[9] + 0 * losses[10] + 0 * losses[11])
        
    
    loss.backward()
    optimizer.step()
    return losses


def train_step_2(model, x, y, other_variable, optimizer, epoch, 
                pre_train_epoch, penalty_coefficient, case1354,
                w_cost=0.000000001, w_active=1.0, w_reactive=1.0, w_voltage=1.0, w_line=1.0):
    """
    Training step 2 - trains encoder constraint losses.
    
    Focuses on constraint satisfaction: active, reactive, voltage, line limits.
    
    Args:
        model: ACOPF model
        x: Ground truth actions
        y: States
        other_variable: Auxiliary variables
        optimizer: Optimizer
        epoch: Current epoch
        pre_train_epoch: Number of pre-training epochs
        penalty_coefficient: Penalty coefficient
        case1354: Power system case data
        w_cost: Weight for cost_loss (losses[4])
        w_active: Weight for active_loss (losses[5])
        w_reactive: Weight for reactive_loss (losses[6])
        w_voltage: Weight for voltage_loss (losses[7])
        w_line: Weight for line_loss (losses[8])
    
    Returns:
        tuple: Loss values
    """
    optimizer.zero_grad()
    losses = compute_loss(model, x, y, other_variable, epoch, 
                         pre_train_epoch, penalty_coefficient, case1354)
    
    # Train encoder constraints
    # Focus on: cost_loss, active_loss, reactive_loss, voltage_loss, line_loss (with configurable weights)
    loss = (0 * losses[0] + 0 * losses[1] + 0 * losses[2] + 0 * losses[3] + 
            w_cost * losses[4] + w_active * losses[5] + w_reactive * losses[6] + 
            w_voltage * losses[7] + w_line * losses[8] + 
            0 * losses[9] + 0 * losses[10] + 0 * losses[11])
    
    loss.backward()
    optimizer.step()
    return losses
