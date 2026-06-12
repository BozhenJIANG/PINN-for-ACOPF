"""
Loss computation for ACOPF training.
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.power_flow import power_flow_equations_batch, power_flow_equations_evaluation


def compute_loss(model, x, y, other_variable, epoch, pre_train_epoch, 
                penalty_coefficient, case1354):
    """
    Compute training losses for ACOPF model.
    
    Args:
        model: ACOPF model
        x: Ground truth actions
        y: States
        other_variable: Auxiliary variables (q, u, delta, balance_theta)
        epoch: Current training epoch
        pre_train_epoch: Number of pre-training epochs
        penalty_coefficient: Coefficient for penalty terms
        case1354: Power system case data
    
    Returns:
        tuple: Various loss components
    """
    x_ = model.encode(y)
    xent_loss = torch.mean((x_ - x)**2)

    if epoch < pre_train_epoch and other_variable is not None:
        # Supervised pre-training: use ground truth q/u/delta to train PINN_PF
        q_u_delta = model.pinn_pf(x, y)
        q_u_delta_loss = torch.mean((q_u_delta - other_variable)**2)

        # temp_p, temp_q = power_flow_equations_evaluation(case1354,  y.detach().cpu().numpy()[0, :], x_.detach().cpu().numpy()[0, :], (model.pinn_pf(x_, y)[0, :]).detach().cpu().numpy())

        # print("training data power flow: ", temp_p, temp_q)


        

        # temp_p, temp_q = power_flow_equations_evaluation(case1354,  y.cpu().numpy()[0, :], x.cpu().numpy()[0, :], other_variable[0, :].cpu().numpy())

        # print("training data power flow: ", temp_p, temp_q)
        # training data power flow:  7.463407e-05 0.00017600207
        # training data power flow:  7.473792e-05 0.00016742705
        # training data power flow:  0.000106489955 0.00019836152
        # training data power flow:  7.6228585e-05 0.00018128102
        # training data power flow:  5.541769e-05 0.00014896343

        return (xent_loss, q_u_delta_loss)
    else:
        # Physics-informed training (main training or unlabeled fine-tuning)
        q_u_delta = model.pinn_pf(x_, y)
        
        if other_variable is not None:
            # Labeled: use ground truth q, u, delta, balance_theta
            q             = other_variable[:, :260]
            u             = other_variable[:, 260:1354]
            delta         = other_variable[:, 1354:]
            balance_theta = other_variable[:, 1354 + 639:1354 + 640].squeeze()
        else:
            # Unlabeled fine-tuning: use PINN's own predictions as physics reference
            q             = q_u_delta[:, :260].detach()
            u             = q_u_delta[:, 260:1354].detach()
            delta         = q_u_delta[:, 1354:].detach()
            balance_theta = delta[:, 639].detach()
        
        # Compute power flow losses
        (grad_P_balance, grad_Q_balance, grad_theta_balance, grad_cost_loss, 
         grad_active_loss, grad_reactive_loss, grad_voltage_loss, grad_line_loss, 
         grad_mse_q, grad_mse_delta, grad_mse_u) = power_flow_equations_batch(
            case1354, y, x_, q, u, delta, q_u_delta, balance_theta)

        # Return weighted losses
        return (
            penalty_coefficient * xent_loss,
            penalty_coefficient * grad_P_balance.mean(),
            penalty_coefficient * grad_Q_balance.mean(),
            grad_theta_balance.mean(),
            penalty_coefficient * grad_cost_loss.mean(),
            penalty_coefficient * grad_active_loss.mean(),
            penalty_coefficient * grad_reactive_loss.mean(),
            penalty_coefficient * grad_voltage_loss.mean(),
            penalty_coefficient * grad_line_loss.mean(),
            penalty_coefficient * grad_mse_q.mean(),
            penalty_coefficient * grad_mse_delta.mean(),
            penalty_coefficient * grad_mse_u.mean()
        )
