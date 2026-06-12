"""
Evaluation functions for ACOPF model.
"""
import numpy as np
import torch
import sys
import os
import copy
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.power_flow import power_flow_equations_evaluation, AC_optimal_power_flow_equations_evaluation

# Get logger
logger = logging.getLogger(__name__)

try:
    from pypower.api import runpf, ppoption
    PYPOWER_AVAILABLE = True
except ImportError:
    PYPOWER_AVAILABLE = False
    print("Warning: pypower not available. Power flow calculations will be limited.")


def run_power_flow_pypower(case1354, state, action):
    """
    Run power flow calculation using pypower to get q, u, delta from state and action.
    Also updates the slack bus active power in action with the power flow result.
    
    Parameters:
    -----------
    case1354 : dict
        Power system case data
    state : array
        State data [Pd, Qd] for all buses (normalized, needs *100)
    action : array  
        Predicted action [Pg_gen (260), Vm_pv (260)] for generators and PV buses
        
    Returns:
    --------
    q : array
        Generator reactive power (260,)
    u : array
        PQ bus voltage magnitudes (1094,)
    delta : array
        All bus voltage angles (1354,)
    action_updated : array
        Updated action with slack bus active power from power flow result
    success : bool
        Whether power flow calculation succeeded
    """
    if not PYPOWER_AVAILABLE:
        return None, None, None, None, False
    
    # Deep copy case data
    ppc = copy.deepcopy(case1354)
    
    # Set power flow options
    ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)
    
    bus_data = ppc['bus']
    gen_data = ppc['gen']
    num_buses = bus_data.shape[0]
    num_gens = gen_data.shape[0]

    # Update load data (Pd, Qd) - state is normalized, needs *100
    for i in range(num_buses):
        bus_data[i, 2] = state[i] * 100  # Pd
        bus_data[i, 3] = state[i + num_buses] * 100  # Qd
    
    # Get bus types
    bus_types = bus_data[:, 1]
    
    # Update generator active power and voltage magnitude
    gen_index = 0
    vm_index = num_gens  # Vm values start after Pg values in action
    
    for i in range(num_buses):
        if bus_types[i] == 2 or bus_types[i] == 3:  # PV bus or Slack bus
            # Find corresponding generator
            for g_idx in range(num_gens):
                if int(gen_data[g_idx, 0]) == i + 1:  # gen bus number matches
                    gen_data[g_idx, 1] = action[gen_index] * 100  # Pg
                    gen_data[g_idx, 5] = action[vm_index]  # Vm
                    gen_index += 1
                    vm_index += 1
                    break
    
    # Run power flow calculation
    try:
        results, success = runpf(ppc, ppopt)
        
        if not success:
            return None, None, None, None, False
        
        # Extract results
        bus_results = results['bus']
        gen_results = results['gen']
        
        # Extract generator reactive power q (260,)
        q = np.zeros(num_gens)
        for g_idx in range(num_gens):
            q[g_idx] = gen_results[g_idx, 2] / 100  # Qg normalized
        
        # Extract PQ bus voltage magnitudes u (1094,)
        pq_bus_indices = np.where(bus_types == 1)[0]
        u = np.zeros(len(pq_bus_indices))
        for idx, bus_idx in enumerate(pq_bus_indices):
            u[idx] = bus_results[bus_idx, 7]  # Vm
        
        # Extract all bus voltage angles delta (1354,)
        delta = np.zeros(num_buses)
        for i in range(num_buses):
            delta[i] = np.deg2rad(bus_results[i, 8])  # Va in radians
        
        # Update slack bus active power in action
        action_updated = action.copy()
        pg_updated = action[:num_gens].copy()
        
        # Only update slack bus active power
        for g_idx in range(num_gens):
            bus_num = int(gen_data[g_idx, 0])
            bus_idx = bus_num - 1
            if bus_types[bus_idx] == 3:  # Slack bus
                pg_updated[g_idx] = gen_results[g_idx, 1] / 100
                break
        
        action_updated[:num_gens] = pg_updated
        
        return q, u, delta, action_updated, True
        
    except Exception as e:
        print(f"Power flow calculation failed: {e}")
        return None, None, None, None, False


def run_power_flow_pypower_limit(case1354, state, action):
    """
    Run power flow calculation using pypower to get q, u, delta from state and action.
    Also updates the slack bus active power in action with the power flow result.
    
    Parameters:
    -----------
    case1354 : dict
        Power system case data
    state : array
        State data [Pd, Qd] for all buses (normalized, needs *100)
    action : array  
        Predicted action [Pg_gen (260), Vm_pv (260)] for generators and PV buses
        
    Returns:
    --------
    q : array
        Generator reactive power (260,)
    u : array
        PQ bus voltage magnitudes (1094,)
    delta : array
        All bus voltage angles (1354,)
    action_updated : array
        Updated action with slack bus active power from power flow result
    success : bool
        Whether power flow calculation succeeded
    """
    if not PYPOWER_AVAILABLE:
        return None, None, None, None, False
    
    # Deep copy case data
    ppc = copy.deepcopy(case1354)
    
    # Set power flow options
    ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)
    
    bus_data = ppc['bus']
    gen_data = ppc['gen']
    num_buses = bus_data.shape[0]
    num_gens = gen_data.shape[0]

    # ------------------------------------------------------------------
    # Clip action to valid physical bounds BEFORE passing to pypower.
    # Prevents V/|V| division-by-zero and singular Jacobian when the
    # model output is outside the feasible region (e.g. early training).
    # ------------------------------------------------------------------
    action = action.copy()
    p_min = gen_data[:, 9] / 100.0
    p_max = gen_data[:, 8] / 100.0
    action[:num_gens]  = np.clip(action[:num_gens],  p_min, p_max)
    action[num_gens:]  = np.clip(action[num_gens:],  0.90,  1.10)

    # print(action[:num_gens])

    # Update load data (Pd, Qd) - state is normalized, needs *100
    for i in range(num_buses):
        bus_data[i, 2] = state[i] * 100  # Pd
        bus_data[i, 3] = state[i + num_buses] * 100  # Qd
    
    # Get bus types
    bus_types = bus_data[:, 1]
    
    # Update generator active power and voltage magnitude
    gen_index = 0
    vm_index = num_gens  # Vm values start after Pg values in action
    
    for i in range(num_buses):
        if bus_types[i] == 2 or bus_types[i] == 3:  # PV bus or Slack bus
            # Find corresponding generator
            for g_idx in range(num_gens):
                if int(gen_data[g_idx, 0]) == i + 1:  # gen bus number matches
                    gen_data[g_idx, 1] = action[gen_index] * 100  # Pg
                    gen_data[g_idx, 5] = action[vm_index]  # Vm
                    gen_index += 1
                    vm_index += 1
                    break
    
    # Run power flow calculation
    try:
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        results, success = runpf(ppc, ppopt)
        
        if not success:
            return None, None, None, None, False
        
        # Extract results
        bus_results = results['bus']
        gen_results = results['gen']
        
        # Extract generator reactive power q (260,)
        q = np.zeros(num_gens)
        for g_idx in range(num_gens):
            q[g_idx] = gen_results[g_idx, 2] / 100  # Qg normalized
        
        # Extract PQ bus voltage magnitudes u (1094,)
        pq_bus_indices = np.where(bus_types == 1)[0]
        u = np.zeros(len(pq_bus_indices))
        for idx, bus_idx in enumerate(pq_bus_indices):
            u[idx] = bus_results[bus_idx, 7]  # Vm
        
        # Extract all bus voltage angles delta (1354,)
        delta = np.zeros(num_buses)
        for i in range(num_buses):
            delta[i] = np.deg2rad(bus_results[i, 8])  # Va in radians
        
        # Update slack bus active power in action
        action_updated = action.copy()
        pg_updated = action[:num_gens].copy()
        
        # Only update slack bus active power
        for g_idx in range(num_gens):
            bus_num = int(gen_data[g_idx, 0])
            bus_idx = bus_num - 1
            if bus_types[bus_idx] == 3:  # Slack bus
                pg_updated[g_idx] = gen_results[g_idx, 1] / 100
                # print("actual pbalance: ", gen_results[g_idx, 1] / 100)
                break
        
        action_updated[:num_gens] = pg_updated
        
        return q, u, delta, action_updated, True
        
    except Exception as e:
        print(f"Power flow calculation failed: {e}")
        return None, None, None, None, False

def validate_model(model, case1354, X_con_test, sample_num_for_validate=50):
    """
    Validate model performance on test data.
    
    Args:
        model: ACOPF model
        case1354: Power system case data
        X_con_test: Test state data
        sample_num_for_validate: Number of samples to validate
    
    Returns:
        tuple: (p_error, q_error) average power flow errors
    """
    model.eval()
    pinn_all_p_error = 0
    pinn_all_q_error = 0
    
    with torch.no_grad():
        # Randomly select samples for validation
        temp_index = np.random.choice(len(X_con_test), sample_num_for_validate, replace=False)
        
        for i in temp_index:
            # Use encoder to predict action
            X_in_pre = model.encode(X_con_test[i:i+1])
            pre_data_pinn = model.pinn_pf(X_in_pre, X_con_test[i:i+1])
            
            # Convert to numpy for calculation (move to CPU first if on GPU)
            state_np = X_con_test[i].cpu().numpy()
            action_np = X_in_pre[0].cpu().numpy()
            q_u_delta_np = pre_data_pinn[0].cpu().numpy()
            
            # Compute power flow error
            temp_p, temp_q = power_flow_equations_evaluation(
                case1354, state_np, action_np, q_u_delta_np)
            
            pinn_all_p_error += temp_p
            pinn_all_q_error += temp_q
    
    return (pinn_all_p_error / sample_num_for_validate, 
            pinn_all_q_error / sample_num_for_validate)


def evaluate_models(model, fc_model, X_con_test_np, X_con_test_tensor, 
                   X_in_test_np, X_in_test_tensor, X_other_test_np,
                   case1354, use_pypower=True):
    """
    Evaluate and compare PINN and FC models.
    
    For PINN and FC models, runs power flow calculation to get q, u, delta
    instead of using ground truth values from test set.
    
    Args:
        model: ACOPF PINN model
        fc_model: Simple FC baseline model
        X_con_test_np: Test state data (numpy)
        X_con_test_tensor: Test state data (tensor)
        X_in_test_np: Test action data (numpy)
        X_in_test_tensor: Test action data (tensor)
        X_other_test_np: Test auxiliary data (numpy)
        case1354: Power system case data
        use_pypower: Whether to use pypower for power flow (if available)
    
    Returns:
        dict: Evaluation metrics for both models and baseline
    """
    model.eval()
    fc_model.eval()
    
    # Get the device from model
    device = next(model.parameters()).device
    
    # Initialize error accumulators
    metrics = {
        'pinn': {'p': 0, 'q': 0, 'cost': 0, 'active': 0, 
                'reactive': 0, 'voltage': 0, 'line': 0},
        'fc': {'p': 0, 'q': 0, 'cost': 0, 'active': 0, 
              'reactive': 0, 'voltage': 0, 'line': 0},
        'baseline': {'p': 0, 'q': 0, 'cost': 0, 'active': 0, 
                    'reactive': 0, 'voltage': 0, 'line': 0}
    }
    
    with torch.no_grad():
        # Move data to the same device as model
        X_con_test_tensor_device = X_con_test_tensor.to(device)
        
        # Get PINN predictions
        action_pinn = model.encode(X_con_test_tensor_device)
        pre_data_pinn = model.pinn_pf(action_pinn, X_con_test_tensor_device)
        
        # Get FC predictions
        pre_data_fc = fc_model(X_con_test_tensor_device)
        
        # Convert to numpy
        action_pinn_np = action_pinn.cpu().numpy()
        pre_data_fc_np = pre_data_fc.cpu().numpy()
    
    # Evaluate each sample
    num_samples = X_con_test_np.shape[0]
    pinn_success_count = 0
    fc_success_count = 0
    
    for i in range(num_samples):
        # ========== PINN evaluation ==========
        # Use pypower to run power flow for PINN prediction
        if use_pypower and PYPOWER_AVAILABLE:
            q_pinn, u_pinn, delta_pinn, action_pinn_updated, pinn_success = \
                run_power_flow_pypower(case1354, X_con_test_np[i, :], action_pinn_np[i, :])
            
            if pinn_success:
                pinn_success_count += 1
                temp_p, temp_q, temp_cost, temp_active, temp_reactive, temp_voltage, temp_line = \
                    AC_optimal_power_flow_equations_evaluation(
                        case1354, X_con_test_np[i, :], action_pinn_updated, 
                        q_pinn, u_pinn, delta_pinn
                    )
            else:
                # Power flow failed, use PINN predicted values directly
                q_pinn = pre_data_pinn[i, :260].cpu().numpy()
                u_pinn = pre_data_pinn[i, 260:1354].cpu().numpy()
                delta_pinn = pre_data_pinn[i, 1354:].cpu().numpy()
                temp_p, temp_q, temp_cost, temp_active, temp_reactive, temp_voltage, temp_line = \
                    AC_optimal_power_flow_equations_evaluation(
                        case1354, X_con_test_np[i, :], action_pinn_np[i, :], 
                        q_pinn, u_pinn, delta_pinn
                    )
        else:
            # Fallback: use PINN predicted values directly
            q_pinn = pre_data_pinn[i, :260].cpu().numpy()
            u_pinn = pre_data_pinn[i, 260:1354].cpu().numpy()
            delta_pinn = pre_data_pinn[i, 1354:].cpu().numpy()
            temp_p, temp_q, temp_cost, temp_active, temp_reactive, temp_voltage, temp_line = \
                AC_optimal_power_flow_equations_evaluation(
                    case1354, X_con_test_np[i, :], action_pinn_np[i, :], 
                    q_pinn, u_pinn, delta_pinn
                )
        
        metrics['pinn']['p'] += np.sqrt(temp_p)
        metrics['pinn']['q'] += np.sqrt(temp_q)
        metrics['pinn']['cost'] += temp_cost
        metrics['pinn']['active'] += temp_active
        metrics['pinn']['reactive'] += temp_reactive
        metrics['pinn']['voltage'] += temp_voltage
        metrics['pinn']['line'] += temp_line

        # ========== FC evaluation ==========
        # Use pypower to run power flow for FC prediction
        if use_pypower and PYPOWER_AVAILABLE:
            q_fc, u_fc, delta_fc, action_fc_updated, fc_success = \
                run_power_flow_pypower(case1354, X_con_test_np[i, :], pre_data_fc_np[i, :])
            
            if fc_success:
                fc_success_count += 1
                temp_p, temp_q, temp_cost, temp_active, temp_reactive, temp_voltage, temp_line = \
                    AC_optimal_power_flow_equations_evaluation(
                        case1354, X_con_test_np[i, :], action_fc_updated,
                        q_fc, u_fc, delta_fc
                    )
            else:
                # Power flow failed, skip this sample for FC
                print(f"Warning: FC power flow failed for sample {i}")
                continue
        else:
            # Fallback: use ground truth (this is incorrect but for compatibility)
            temp_p, temp_q, temp_cost, temp_active, temp_reactive, temp_voltage, temp_line = \
                AC_optimal_power_flow_equations_evaluation(
                    case1354, X_con_test_np[i, :], pre_data_fc_np[i, :],
                    X_other_test_np[i, :260], X_other_test_np[i, 260:1354], 
                    X_other_test_np[i, 1354:])
        
        metrics['fc']['p'] += np.sqrt(temp_p)
        metrics['fc']['q'] += np.sqrt(temp_q)
        metrics['fc']['cost'] += temp_cost
        metrics['fc']['active'] += temp_active
        metrics['fc']['reactive'] += temp_reactive
        metrics['fc']['voltage'] += temp_voltage
        metrics['fc']['line'] += temp_line

        # ========== Baseline (true values) evaluation ==========
        # Baseline uses ground truth values from test set
        temp_p, temp_q, temp_cost, temp_active, temp_reactive, temp_voltage, temp_line = \
            AC_optimal_power_flow_equations_evaluation(
                case1354, X_con_test_np[i, :], X_in_test_np[i, :],
                X_other_test_np[i, :260], X_other_test_np[i, 260:1354], 
                X_other_test_np[i, 1354:])
        metrics['baseline']['p'] += np.sqrt(temp_p)
        metrics['baseline']['q'] += np.sqrt(temp_q)
        metrics['baseline']['cost'] += temp_cost
        metrics['baseline']['active'] += temp_active
        metrics['baseline']['reactive'] += temp_reactive
        metrics['baseline']['voltage'] += temp_voltage
        metrics['baseline']['line'] += temp_line

    # Average errors
    for model_name in metrics:
        for key in metrics[model_name]:
            metrics[model_name][key] /= num_samples
    
    # Print and log power flow success rates
    if use_pypower and PYPOWER_AVAILABLE:
        success_msg = f"\nPower Flow Success Rates:\n"
        success_msg += f"  PINN: {pinn_success_count}/{num_samples} ({100*pinn_success_count/num_samples:.1f}%)\n"
        success_msg += f"  FC: {fc_success_count}/{num_samples} ({100*fc_success_count/num_samples:.1f}%)"
        print(success_msg)
        logger.info(success_msg)

    return metrics


def print_evaluation_results(metrics):
    """
    Print and log evaluation results in formatted table.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    # Build output string
    output = "\n" + "="*80 + "\n"
    output += "ACOPF Equation Evaluation Results\n"
    output += "="*80
    
    print(output)
    logger.info(output)
    
    for model_name, model_metrics in metrics.items():
        model_output = f"\n{model_name.upper()} Model:\n"
        model_output += f"  P Error: {model_metrics['p']:.15f}\n"
        model_output += f"  Q Error: {model_metrics['q']:.15f}\n"
        model_output += f"  Cost Error: {model_metrics['cost']:.15f}\n"
        model_output += f"  Active Power Limit Error: {model_metrics['active']:.15f}\n"
        model_output += f"  Reactive Power Limit Error: {model_metrics['reactive']:.15f}\n"
        model_output += f"  Voltage Limit Error: {model_metrics['voltage']:.15f}\n"
        model_output += f"  Line Limit Error: {model_metrics['line']:.15f}"
        
        # print(model_output)
        logger.info(model_output)
