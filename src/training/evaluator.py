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
