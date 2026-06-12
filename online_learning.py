#!/usr/bin/env python3
"""
Online Learning Framework for ACOPF with Data Shift.

Only for small load distribution and topology change scenarios

This script implements:
1. Pre-training on labeled data (1000 samples)
2. Online fine-tuning on unlabeled data with physics constraints
3. Evaluation on labeled test data for both load and topology variations
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.online_dataset import (
    prepare_online_learning_data, create_tensors, create_dataloaders
)
from src.models.gnn_encoder import GNNACOPFModel
from src.training.trainer import train_step_0, train_step_1_1, train_step_2
from src.training.evaluator import run_power_flow_pypower, PYPOWER_AVAILABLE, validate_model
from src.utils.power_flow import calculate_ybus, AC_optimal_power_flow_equations_evaluation,power_flow_equations_evaluation
from src.utils import update_learning_rate
from case1354 import case1354


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Online Learning for ACOPF')
    
    # Model architecture
    parser.add_argument('--intermediate_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--gnn_hidden_per_bus', type=int, default=4,
                        help='GNN hidden features per bus node')
    
    # Training parameters
    parser.add_argument('--pretrain_epochs', type=int, default=1000)
    parser.add_argument('--finetune_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--betas', type=float, nargs=2, default=[0.8, 0.9])
    
    # Loss weights
    parser.add_argument('--w_p_balance', type=float, default=1.0)
    parser.add_argument('--w_q_balance', type=float, default=1.0)
    parser.add_argument('--w_theta_balance', type=float, default=1.0)
    parser.add_argument('--w_cost', type=float, default=1e-9)
    parser.add_argument('--w_active', type=float, default=1.0)
    parser.add_argument('--w_reactive', type=float, default=1.0)
    parser.add_argument('--w_voltage', type=float, default=1.0)
    parser.add_argument('--w_line', type=float, default=1.0)
    
    # Training control
    parser.add_argument('--step_pretrain_batches', type=int, default=1)
    parser.add_argument('--step_pinn_batches', type=int, default=4)
    parser.add_argument('--step_encoder_batches', type=int, default=1)
    parser.add_argument('--penalty_coefficient', type=float, default=1.0)
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='./CompleteDataSet')
    parser.add_argument('--save_dir', type=str, default='./save_model_online')
    parser.add_argument('--results_dir', type=str, default='./results_online')
    
    # Device
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cpu', action='store_true')
    
    # Modes
    parser.add_argument('--skip_pretrain', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--skip_load_variation', action='store_true')
    parser.add_argument('--skip_topology_variation', action='store_true')
    
    # Validation
    parser.add_argument('--validate_every', type=int, default=100,
                        help='Run validation every N epochs')
    parser.add_argument('--sample_num_for_validate', type=int, default=20,
                        help='Number of test samples used for intermediate validation')
    parser.add_argument('--finetune_label_ratio', type=float, default=0.0,
                        help='Fraction of 2000 finetune samples to use as labeled '
                             '(e.g. 0.01=20, 0.05=100, 0.1=200, 0.2=400). '
                             'Remaining samples use unsupervised physics loss.')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Fixed output CSV path for results. '
                             'If not set, a timestamped filename is generated automatically.')
    
    return parser.parse_args()


def setup_logging(log_dir):
    """Setup logging."""
    import logging
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'online_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def pretrain_model(model, train_loader, X_con_val_tensor, args, device, case, logger):
    """Pre-train model on labeled training data with intermediate train-set validation."""
    logger.info("="*80)
    logger.info("Phase 1: Pre-training on labeled data")
    logger.info("="*80)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr, betas=tuple(args.betas)
    )
    
    best_p_error = float('inf')
    save_path = os.path.join(args.save_dir, 'pretrained_model.pth')
    
    tqdm_e = tqdm(range(1, args.pretrain_epochs + 1), desc='Pre-training', leave=True)
    for epoch in tqdm_e:
        model.train()
        total_loss  = 0
        batch_count = 0
        penalty_coefficient = args.penalty_coefficient

        # Update learning rate (same schedule as main.py)
        new_lr = update_learning_rate(optimizer, epoch, args.lr)

        for batch_idx, (train_x, train_y, other_var) in enumerate(train_loader):
            if batch_idx >= args.step_pretrain_batches:
                break
            train_x   = train_x.to(device)
            train_y   = train_y.to(device)
            other_var = other_var.to(device)
            losses = train_step_0(
                model, train_y, train_x, other_var,
                optimizer, epoch, args.pretrain_epochs,
                penalty_coefficient, case
            )
            total_loss  += losses[0].item()
            batch_count += 1

        # # for test
        # for batch_idx, (train_x, train_y, other_variable) in enumerate(train_loader):
        #     print(train_x.shape[0])
        #     temp_p_, temp_q_ = 0, 0
        #     for i in range(train_x.shape[0]):


        #         temp_p, temp_q = power_flow_equations_evaluation(case,  train_x.cpu().numpy()[i, :], train_y.cpu().numpy()[i, :], other_variable[i, :].cpu().numpy())

        #         temp_p_ += np.sqrt(temp_p)
        #         temp_q_ += np.sqrt(temp_q)

        #     break
        # print("training data power flow: ", temp_p_/train_x.shape[0],temp_q_/train_x.shape[0])
        # # 128
        # # training data power flow:  0.008188682 0.012491809

        # print(losses[0]," ",losses[1])

        avg_loss = total_loss / batch_count if batch_count > 0 else 0

        # Intermediate validation on training set
        if epoch % args.validate_every == 0 or epoch == 1:
            X_con_val_dev = X_con_val_tensor.to(device)
            p_error, q_error = validate_model(
                model, case, X_con_val_dev, args.sample_num_for_validate
            )

            logger.info(
                f'Epoch {epoch}/{args.pretrain_epochs} | LR: {new_lr:.2e} | Loss: {avg_loss:.6f} | '
                f'P-err: {p_error:.4f} | Q-err: {q_error:.4f}'
            )
            if p_error < best_p_error:
                best_p_error = p_error
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({'model': model.state_dict(), 'epoch': epoch,
                            'p_error': best_p_error}, save_path)
            tqdm_e.set_description(
                f'Pre-training | Loss: {avg_loss:.4f} | '
                f'P-err: {p_error:.4f} | Best-P: {best_p_error:.4f}'
            )
            tqdm_e.set_postfix({'LR': f'{new_lr:.2e}'})
        else:
            tqdm_e.set_description(f'Pre-training | Loss: {avg_loss:.4f}')
            tqdm_e.set_postfix({'LR': f'{new_lr:.2e}'})
    
    logger.info(f"Pre-training completed. Best P error: {best_p_error:.6f}")
    logger.info(f"Model saved to: {save_path}")
    
    return model


def finetune_model(model, finetune_data, args, device, case, logger):
    """Fine-tune model using a mix of labeled (supervised) and unlabeled (physics) data.

    finetune_data : dict with keys X_con, X_in, X_other for all N (2000) samples.

    Splitting controlled by args.finetune_label_ratio  (p):
      - first  int(p * N) samples → supervised   via train_step_0 (labeled)
      - last (1-p)*N samples      → unsupervised via train_step_1_1 + train_step_2

    p = 0.0 (default) → fully unsupervised (original behaviour)
    """
    logger.info("="*80)
    logger.info("Phase 2: Fine-tuning on unlabeled data")
    logger.info("="*80)

    X_con_ft   = finetune_data['X_con']
    X_in_ft    = finetune_data['X_in']
    X_other_ft = finetune_data['X_other']
    N = X_con_ft.shape[0]   # typically 2000

    p           = args.finetune_label_ratio
    n_labeled   = max(0, min(N, int(p * N)))
    n_unlabeled = N - n_labeled
    logger.info(f"  Finetune split: {n_labeled} labeled ({p*100:.1f}%) "
                f"+ {n_unlabeled} unlabeled ({(1-p)*100:.1f}%)")

    # Labeled loader  (X_con, X_in, X_other) — for supervised train_step_0
    labeled_loader = None
    if n_labeled > 0:
        labeled_loader = create_dataloaders(
            X_con_ft[:n_labeled], X_in_ft[:n_labeled], X_other_ft[:n_labeled],
            batch_size=args.batch_size, shuffle=True
        )

    # Unlabeled loader (X_con only) — for physics-informed train_step_1_1 + train_step_2
    unlabeled_loader = None
    if n_unlabeled > 0:
        unlabeled_loader = create_dataloaders(
            X_con_ft[n_labeled:],
            batch_size=args.batch_size, shuffle=True
        )

    # Build a small validation tensor from finetune data (always available)
    val_tensor = torch.tensor(
        X_con_ft[:args.sample_num_for_validate], dtype=torch.float32
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr, betas=tuple(args.betas)
    )

    best_p_error = float('inf')
    # pseudo_epoch ensures loss.py always enters the physics branch
    # AND drives the LR schedule to continue from where pretrain left off
    pseudo_epoch_base = args.pretrain_epochs

    tqdm_e = tqdm(range(1, args.finetune_epochs + 1), desc='Fine-tuning', leave=True)
    for epoch in tqdm_e:
        model.train()
        total_loss  = 0
        batch_count = 0
        pseudo_epoch = pseudo_epoch_base + epoch
        penalty_coefficient = args.penalty_coefficient

        # Update LR using pseudo_epoch so schedule continues from pretrain end
        new_lr = update_learning_rate(optimizer, pseudo_epoch, args.lr)
        
        # ---- Step 0: supervised fine-tuning on labeled subset ----
        if labeled_loader is not None:
            
            for batch_idx, (lx, ly, lo) in enumerate(labeled_loader):
                # if batch_idx >= 4 * args.step_pretrain_batches:
                if batch_idx >= args.step_pretrain_batches:
                    break
                lx = lx.to(device)
                ly = ly.to(device)
                lo = lo.to(device)
                losses = train_step_0(
                    model, ly, lx, lo,
                    optimizer, pseudo_epoch, args.pretrain_epochs,
                    penalty_coefficient, case
                )
                total_loss  += losses[0].item()
                batch_count += 1             

        # ---- Step 1.1: PINN_PF on unlabeled subset ----
        if unlabeled_loader is not None:

            for batch_idx, (unlabeled_x,) in enumerate(unlabeled_loader):
                if batch_idx >= args.step_pinn_batches:
                    break
                unlabeled_x = unlabeled_x.to(device)
                with torch.no_grad():
                    pseudo_action = model.encode(unlabeled_x)
                losses = train_step_1_1(
                    model, pseudo_action, unlabeled_x, None,
                    optimizer, pseudo_epoch, args.pretrain_epochs,
                    penalty_coefficient, case,
                    args.w_p_balance, args.w_q_balance, args.w_theta_balance
                )
                total_loss  += losses[0].item()
                batch_count += 1

        # ---- Step 2: encoder constraints on unlabeled subset ----
        if unlabeled_loader is not None:
            # Freeze PINN_PF parameters so only encoder gets updated
            for param in model.pinn_pf_.parameters():
                param.requires_grad = False
            for batch_idx, (unlabeled_x,) in enumerate(unlabeled_loader):
                if batch_idx >= args.step_encoder_batches:
                    # Unfreeze PINN_PF parameters for subsequent steps
                    for param in model.pinn_pf_.parameters():
                        param.requires_grad = True
                    break
                unlabeled_x = unlabeled_x.to(device)
                # with torch.no_grad():
                pseudo_action = model.encode(unlabeled_x)
                losses = train_step_2(
                    model, pseudo_action, unlabeled_x, None,
                    optimizer, pseudo_epoch, args.pretrain_epochs,
                    penalty_coefficient, case,
                    args.w_cost, args.w_active, args.w_reactive,
                    args.w_voltage, args.w_line
                )
                total_loss  += losses[0].item()
                batch_count += 1


        avg_loss = total_loss / batch_count if batch_count > 0 else 0

        # Intermediate validation on test set
        if epoch % args.validate_every == 0 or epoch == 1:
            X_con_test_dev = val_tensor.to(device)
            p_error, q_error = validate_model(
                model, case, X_con_test_dev, args.sample_num_for_validate
            )
            logger.info(
                f'Epoch {epoch}/{args.finetune_epochs} | LR: {new_lr:.2e} | Loss: {avg_loss:.6f} | '
                f'P-err: {p_error:.4f} | Q-err: {q_error:.4f}'
            )
            if p_error < best_p_error:
                best_p_error = p_error
            tqdm_e.set_description(
                f'Fine-tuning | Loss: {avg_loss:.4f} | '
                f'P-err: {p_error:.4f} | Best-P: {best_p_error:.4f}'
            )
            tqdm_e.set_postfix({'LR': f'{new_lr:.2e}'})
        else:
            tqdm_e.set_description(f'Fine-tuning | Loss: {avg_loss:.4f}')
            tqdm_e.set_postfix({'LR': f'{new_lr:.2e}'})

    logger.info(f"Fine-tuning completed. Best P error: {best_p_error:.6f}")

    return model

def finetune_model(model, finetune_data, args, device, case, logger):
    """Fine-tune model using a mix of labeled (supervised) and unlabeled (physics) data.

    finetune_data : dict with keys X_con, X_in, X_other for all N (2000) samples.

    Splitting controlled by args.finetune_label_ratio  (p):
      - first  int(p * N) samples → supervised   via train_step_0 (labeled)
      - last (1-p)*N samples      → unsupervised via train_step_1_1 + train_step_2

    p = 0.0 (default) → fully unsupervised (original behaviour)
    """
    logger.info("="*80)
    logger.info("Phase 2: Fine-tuning on unlabeled data")
    logger.info("="*80)

    X_con_ft   = finetune_data['X_con']
    X_in_ft    = finetune_data['X_in']
    X_other_ft = finetune_data['X_other']
    N = X_con_ft.shape[0]   # typically 2000

    p           = args.finetune_label_ratio
    n_labeled   = max(0, min(N, int(p * N)))
    n_unlabeled = N - n_labeled
    logger.info(f"  Finetune split: {n_labeled} labeled ({p*100:.1f}%) "
                f"+ {n_unlabeled} unlabeled ({(1-p)*100:.1f}%)")

    # Labeled loader  (X_con, X_in, X_other) — for supervised train_step_0
    labeled_loader = None
    if n_labeled > 0:
        labeled_loader = create_dataloaders(
            X_con_ft[:n_labeled], X_in_ft[:n_labeled], X_other_ft[:n_labeled],
            batch_size=args.batch_size, shuffle=True
        )

    # Unlabeled loader (X_con only) — for physics-informed train_step_1_1 + train_step_2
    unlabeled_loader = None
    if n_unlabeled > 0:
        unlabeled_loader = create_dataloaders(
            X_con_ft[n_labeled:],
            batch_size=args.batch_size, shuffle=True
        )

    # Build a small validation tensor from finetune data (always available)
    val_tensor = torch.tensor(
        X_con_ft[:args.sample_num_for_validate], dtype=torch.float32
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr, betas=tuple(args.betas)
    )

    # ---- Step 0: supervised fine-tuning on labeled subset ----
    if labeled_loader is not None:   
        best_p_error = float('inf')     
        tqdm_e = tqdm(range(1, args.pretrain_epochs + 1), desc='Pre-training', leave=True)
        for epoch in tqdm_e:
            model.train()
            total_loss  = 0
            batch_count = 0
            penalty_coefficient = args.penalty_coefficient

            # Update learning rate (same schedule as main.py)
            new_lr = update_learning_rate(optimizer, epoch, args.lr)

            for batch_idx, (train_x, train_y, other_var) in enumerate(labeled_loader):
                if batch_idx >= args.step_pretrain_batches:
                    break
                train_x   = train_x.to(device)
                train_y   = train_y.to(device)
                other_var = other_var.to(device)
                losses = train_step_0(
                    model, train_y, train_x, other_var,
                    optimizer, epoch, args.pretrain_epochs,
                    penalty_coefficient, case
                )
                total_loss  += losses[0].item()
                batch_count += 1

            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            # Intermediate validation on test set
            if epoch % args.validate_every == 0 or epoch == 1:
                X_con_test_dev = val_tensor.to(device)
                p_error, q_error = validate_model(
                    model, case, X_con_test_dev, args.sample_num_for_validate
                )
                logger.info(
                    f'Epoch {epoch}/{args.finetune_epochs} | LR: {new_lr:.2e} | Loss: {avg_loss:.6f} | '
                    f'P-err: {p_error:.4f} | Q-err: {q_error:.4f}'
                )
                if p_error < best_p_error:
                    best_p_error = p_error
                tqdm_e.set_description(
                    f'Fine-tuning | Loss: {avg_loss:.4f} | '
                    f'P-err: {p_error:.4f} | Best-P: {best_p_error:.4f}'
                )
                tqdm_e.set_postfix({'LR': f'{new_lr:.2e}'})
            else:
                tqdm_e.set_description(f'Fine-tuning | Loss: {avg_loss:.4f}')
                tqdm_e.set_postfix({'LR': f'{new_lr:.2e}'})



    best_p_error = float('inf')
    pseudo_epoch_base = args.pretrain_epochs
    # pseudo_epoch ensures loss.py always enters the physics branch
    # AND drives the LR schedule to continue from where pretrain left off
    tqdm_e = tqdm(range(1, args.finetune_epochs + 1), desc='Fine-tuning', leave=True)
    for epoch in tqdm_e:
        model.train()
        total_loss  = 0
        batch_count = 0  
        pseudo_epoch = pseudo_epoch_base + epoch
        penalty_coefficient = args.penalty_coefficient

        # Update LR using pseudo_epoch so schedule continues from pretrain end
        new_lr = update_learning_rate(optimizer, pseudo_epoch, args.lr)

        # ---- Step 1.1: PINN_PF on unlabeled subset ----
        if unlabeled_loader is not None:
            for batch_idx, (unlabeled_x,) in enumerate(unlabeled_loader):
                if batch_idx >= args.step_pinn_batches:
                    break
                unlabeled_x = unlabeled_x.to(device)
                with torch.no_grad():
                    pseudo_action = model.encode(unlabeled_x)
                losses = train_step_1_1(
                    model, pseudo_action, unlabeled_x, None,
                    optimizer, pseudo_epoch, args.pretrain_epochs,
                    penalty_coefficient, case,
                    args.w_p_balance, args.w_q_balance, args.w_theta_balance
                )
                total_loss  += losses[0].item()
                batch_count += 1

        # ---- Step 2: encoder constraints on unlabeled subset ----
        if unlabeled_loader is not None:
            # Freeze PINN_PF parameters so only encoder gets updated
            for param in model.pinn_pf_.parameters():
                param.requires_grad = False
            for batch_idx, (unlabeled_x,) in enumerate(unlabeled_loader):
                if batch_idx >= args.step_encoder_batches:
                    # Unfreeze PINN_PF parameters for subsequent steps
                    for param in model.pinn_pf_.parameters():
                        param.requires_grad = True
                    break
                unlabeled_x = unlabeled_x.to(device)
                # with torch.no_grad():
                pseudo_action = model.encode(unlabeled_x)
                losses = train_step_2(
                    model, pseudo_action, unlabeled_x, None,
                    optimizer, pseudo_epoch, args.pretrain_epochs,
                    penalty_coefficient, case,
                    args.w_cost, args.w_active, args.w_reactive,
                    args.w_voltage, args.w_line
                )
                total_loss  += losses[0].item()
                batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0

        # Intermediate validation on test set
        if epoch % args.validate_every == 0 or epoch == 1:
            X_con_test_dev = val_tensor.to(device)
            p_error, q_error = validate_model(
                model, case, X_con_test_dev, args.sample_num_for_validate
            )
            logger.info(
                f'Epoch {epoch}/{args.finetune_epochs} | LR: {new_lr:.2e} | Loss: {avg_loss:.6f} | '
                f'P-err: {p_error:.4f} | Q-err: {q_error:.4f}'
            )
            if p_error < best_p_error:
                best_p_error = p_error
            tqdm_e.set_description(
                f'Fine-tuning | Loss: {avg_loss:.4f} | '
                f'P-err: {p_error:.4f} | Best-P: {best_p_error:.4f}'
            )
            tqdm_e.set_postfix({'LR': f'{new_lr:.2e}'})
        else:
            tqdm_e.set_description(f'Fine-tuning | Loss: {avg_loss:.4f}')
            tqdm_e.set_postfix({'LR': f'{new_lr:.2e}'})

    logger.info(f"Fine-tuning completed. Best P error: {best_p_error:.6f}")

    return model

def evaluate_on_test(model, test_data, case, device, logger):
    """
    Evaluate GNN model on labeled test data using physics-based metrics
    consistent with main.py: P/Q power flow error, cost, active/reactive/
    voltage/line constraint violations.
    """
    model.eval()

    # for test
    # X_con_np  = test_data['X_con'][:5,:]   # (N, 2708) states
    # X_in_np   = test_data['X_in'][:5,:]    # (N, 520)  actions (ground truth)
    # X_other_np = test_data['X_other'][:5,:] # (N, 2708) q|u|delta ground truth

    X_con_np  = test_data['X_con']  # (N, 2708) states
    X_in_np   = test_data['X_in']    # (N, 520)  actions (ground truth)
    X_other_np = test_data['X_other'] # (N, 2708) q|u|delta ground truth

    X_con_tensor = torch.tensor(X_con_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        action_pred = model.encode(X_con_tensor)          # [N, 520]
        pf_pred     = model.pinn_pf(action_pred, X_con_tensor)  # [N, 2708]

    action_pred_np = action_pred.cpu().numpy()
    pf_pred_np     = pf_pred.cpu().numpy()

    num_samples = X_con_np.shape[0]
    keys = ['p', 'q', 'cost', 'active', 'reactive', 'voltage', 'line']
    acc  = {k: 0.0 for k in keys}
    acc_baseline = {k: 0.0 for k in keys}
    success_count = 0

    for i in range(num_samples):
        # ---- Try pypower power flow for accurate q/u/delta ----
        if PYPOWER_AVAILABLE:
            q, u, delta, action_updated, ok = run_power_flow_pypower(
                case, X_con_np[i], action_pred_np[i])
            if ok:
                success_count += 1
            else:
                # Fallback: use PINN predictions
                q     = pf_pred_np[i, :260]
                u     = pf_pred_np[i, 260:1354]
                delta = pf_pred_np[i, 1354:]
                action_updated = action_pred_np[i]
        else:
            q     = pf_pred_np[i, :260]
            u     = pf_pred_np[i, 260:1354]
            delta = pf_pred_np[i, 1354:]
            action_updated = action_pred_np[i]

        p_err, q_err, cost, active, reactive, voltage, line = \
            AC_optimal_power_flow_equations_evaluation(
                case, X_con_np[i], action_updated, q, u, delta)

        acc['p']        += np.sqrt(p_err)
        acc['q']        += np.sqrt(q_err)
        acc['cost']     += cost
        acc['active']   += active
        acc['reactive'] += reactive
        acc['voltage']  += voltage
        acc['line']     += line

        # ---- Baseline: ground truth values ----
        p_err, q_err, cost, active, reactive, voltage, line = \
            AC_optimal_power_flow_equations_evaluation(
                case, X_con_np[i], X_in_np[i],
                X_other_np[i, :260],
                X_other_np[i, 260:1354],
                X_other_np[i, 1354:])
        acc_baseline['p']        += np.sqrt(p_err)
        acc_baseline['q']        += np.sqrt(q_err)
        acc_baseline['cost']     += cost
        acc_baseline['active']   += active
        acc_baseline['reactive'] += reactive
        acc_baseline['voltage']  += voltage
        acc_baseline['line']     += line

    for k in keys:
        acc[k]          /= num_samples
        acc_baseline[k] /= num_samples

    if PYPOWER_AVAILABLE:
        logger.info(f"  Power Flow Success: {success_count}/{num_samples} "
                    f"({100*success_count/num_samples:.1f}%)")

    logger.info("  --- GNN Prediction ---")
    for k in keys:
        logger.info(f"    {k:>10}: {acc[k]:.8f}")
    logger.info("  --- Baseline (Ground Truth) ---")
    for k in keys:
        logger.info(f"    {k:>10}: {acc_baseline[k]:.8f}")

    # Flat dict for CSV saving
    metrics = {f'pred_{k}': acc[k] for k in keys}
    metrics.update({f'baseline_{k}': acc_baseline[k] for k in keys})
    pf_sr = success_count / num_samples if PYPOWER_AVAILABLE else float('nan')
    metrics['pred_pf_success_rate'] = pf_sr
    return metrics


def _make_model_copy(model, args, device):
    """Create a fresh GNNACOPFModel copy with pre-trained weights."""
    copy = GNNACOPFModel(
        model.state_dim, model.act_dim,
        args.intermediate_dim, args.latent_dim,
        model.num_buses,
        model.encoder.ybus.cpu().numpy(),
        model.limits, model.limits_q,
        gnn_hidden_per_bus=args.gnn_hidden_per_bus
    ).to(device)
    copy.load_state_dict(model.state_dict())
    return copy


def make_topology_case(base_case, line_idx):
    """Return a deep copy of base_case with branch line_idx (0-based) set offline.

    The scenario names '9','7','20','19','17' are treated as 0-based indices into
    case['branch'].  Set BR_STATUS (column 10) to 0 to disconnect the line.
    """
    import copy as _copy
    topo_case = _copy.deepcopy(base_case)
    topo_case['branch'][line_idx - 1, 10] = 0   # BR_STATUS = 0 → line offline
    return topo_case


def _make_topology_model_copy(model, topo_case, args, device):
    """Like _make_model_copy but overwrites adj_matrix buffers to reflect the
    modified topology (disconnected line) so GNN message-passing uses the correct
    graph structure for topology-change scenarios.
    """
    model_copy = _make_model_copy(model, args, device)   # loads all weights + original buffers

    # Recompute binary adjacency from the modified Ybus (returns a complex Tensor)
    ybus_topo  = calculate_ybus(
        topo_case['branch'], topo_case['bus'].shape[0], topo_case['bus']
    )                                                    # complex Tensor [N, N]
    adj_tensor = (torch.abs(ybus_topo) > 0).float().to(device)  # binary {0,1}

    # Overwrite the adjacency buffers in both sub-modules
    model_copy.encoder.adj_matrix.copy_(adj_tensor)
    model_copy.pinn_pf_.adj_matrix.copy_(adj_tensor)

    return model_copy


def run_load_variation_experiments(model, data, args, device, case, logger):
    """Run experiments for load variation scenarios."""
    logger.info("="*80)
    logger.info("Load Variation Experiments")
    logger.info("="*80)
    
    results = []
    
    for scenario_name, scenario_data in data['load_variation'].items():
        # if scenario_name == "0.01":
        logger.info(f"\nScenario: Load shift = {scenario_name}")
        
        model_copy = _make_model_copy(model, args, device)
        model_copy = finetune_model(
            model_copy, scenario_data['finetune_data'], args, device, case, logger
        )
        test_metrics = evaluate_on_test(model_copy, scenario_data['test_labeled'], case, device, logger)
        
        results.append({
            'scenario_type': 'load_variation',
            'scenario_name': scenario_name,
            **test_metrics
        })
    
    return pd.DataFrame(results)


def run_topology_variation_experiments(model, data, args, device, case, logger):
    """Run experiments for topology variation scenarios.

    For each scenario a modified case (line disconnected) is built and used for:
      - GNN adjacency matrix update (correct graph structure)
      - physics loss during fine-tuning
      - evaluation metrics
    """
    logger.info("="*80)
    logger.info("Topology Variation Experiments")
    logger.info("="*80)
    
    results = []
    
    for scenario_name, scenario_data in data['topology_change'].items():
        line_idx = int(scenario_name)   # 0-based index into case['branch']
        logger.info(f"\nScenario: Line {scenario_name} (branch[{line_idx}]) disconnected")

        # Build modified case and topology-aware model copy
        topo_case  = make_topology_case(case, line_idx)
        model_copy = _make_topology_model_copy(model, topo_case, args, device)

        model_copy = finetune_model(
            model_copy, scenario_data['finetune_data'], args, device, topo_case, logger
        )
        test_metrics = evaluate_on_test(
            model_copy, scenario_data['test_labeled'], topo_case, device, logger
        )
        
        results.append({
            'scenario_type': 'topology_change',
            'scenario_name': f'line_{scenario_name}',
            **test_metrics
        })
    
    return pd.DataFrame(results)


def main():
    """Main function."""
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    logger = setup_logging(args.results_dir)
    logger.info("Online Learning for ACOPF")
    logger.info(f"Arguments: {vars(args)}")
    
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    logger.info("Loading data...")
    data = prepare_online_learning_data(
        args.data_dir,
        load_variation=not args.skip_load_variation,
        topology_change=not args.skip_topology_variation,
        load_scenarios = ['-0.03', '-0.01', '0.0', '0.01', '0.03']
    )
    
    case = case1354()
    branch_data = case['branch']
    bus_data = case['bus']
    num_buses = bus_data.shape[0]
    ybus_matrix = calculate_ybus(branch_data, num_buses, bus_data)
    
    state_dim = data['labeled_train']['X_con'].shape[1]
    act_dim = data['labeled_train']['X_in'].shape[1]
    
    # Define action limits (generator active power + voltage limits)
    temp_data = case["gen"]
    limits = [(temp_data[i][9]/100, temp_data[i][8]/100) for i in range(len(temp_data))] + \
             [(0.9, 1.1) for _temp_data in case["bus"] if _temp_data[1] in [2, 3]]
    
    # Define reactive power limits from generator data
    limits_q = [(temp_data[i][4]/100, temp_data[i][3]/100) for i in range(len(temp_data))]
    
    logger.info("Initializing GNN model...")
    model = GNNACOPFModel(
        state_dim, act_dim, args.intermediate_dim, args.latent_dim,
        num_buses, ybus_matrix, limits, limits_q,
        gnn_hidden_per_bus=args.gnn_hidden_per_bus
    ).to(device)
        
    if not args.skip_pretrain:
        X_con_train = data['labeled_train']['X_con']
        X_in_train = data['labeled_train']['X_in']
        X_other_train = data['labeled_train']['X_other']
        
        train_loader = create_dataloaders(
            X_con_train, X_in_train, X_other_train,
            batch_size=args.batch_size, shuffle=True
        )
        # Use training set for pretrain validation (same distribution as training data)
        X_con_train_tensor = torch.tensor(X_con_train, dtype=torch.float32)
        model = pretrain_model(model, train_loader, X_con_train_tensor, args, device, case, logger)
    elif args.pretrained_model:
        logger.info(f"Loading pretrained model from {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        model.load_state_dict(checkpoint['model'])
    
    all_results = []

    if not args.skip_load_variation:
        load_results = run_load_variation_experiments(model, data, args, device, case, logger)
        all_results.append(load_results)
    
    if not args.skip_topology_variation:
        topo_results = run_topology_variation_experiments(model, data, args, device, case, logger)
        all_results.append(topo_results)
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        
        if args.results_file:
            results_file = args.results_file
            os.makedirs(os.path.dirname(results_file) or '.', exist_ok=True)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(args.results_dir, f'results_{timestamp}.csv')
        final_results.to_csv(results_file, index=False)
        logger.info(f"\nResults saved to: {results_file}")
        
        logger.info("\n" + "="*80)
        logger.info("Results Summary")
        logger.info("="*80)
        logger.info(final_results.to_string())
    
    logger.info("\nOnline learning experiments completed!")


if __name__ == '__main__':
    main()
