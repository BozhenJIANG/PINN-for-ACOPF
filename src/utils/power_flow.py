"""
Power flow equation calculations for training and evaluation.
"""
import numpy as np
import torch
from .power_system import calculate_ybus, calculate_ybus_no_shunt, calculate_ybus_numpy


def power_flow_equations_batch(case1354, state, action, q, u, delta, q_u_delta, balance_theta):
    """
    Batched version of power flow equations for training.
    
    All input tensors have batch_size as first dimension.
    
    Args:
        case1354: Power system case data dictionary
        state: State tensor [batch_size, state_dim]
        action: Action tensor [batch_size, act_dim]
        q: Reactive power reference [batch_size, 260]
        u: Voltage reference [batch_size, 1094]
        delta: Angle reference [batch_size, 1354]
        q_u_delta: Predicted values [batch_size, 2708]
        balance_theta: Balance node angle [batch_size]
    
    Returns:
        tuple: Various power flow losses and metrics
    """
    bus_data = case1354['bus']
    gen_data_ = case1354['gen']
    branch_data = case1354['branch']
    gencost_data = case1354['gencost']

    num_buses = bus_data.shape[0]
    num_gens = gen_data_.shape[0]
    batch_size = state.shape[0]

    # Calculate Ybus (constants, not dependent on batch data)
    Ybus = calculate_ybus(branch_data, num_buses, bus_data)
    Ybus_ = calculate_ybus_no_shunt(branch_data, num_buses, bus_data)
    
    # Ensure Ybus tensors are on the same device as input tensors
    device = state.device
    if isinstance(Ybus, torch.Tensor) and Ybus.device != device:
        Ybus = Ybus.to(device)
    elif isinstance(Ybus, np.ndarray):
        Ybus = torch.tensor(Ybus, dtype=torch.complex64, device=device)
    
    if isinstance(Ybus_, torch.Tensor) and Ybus_.device != device:
        Ybus_ = Ybus_.to(device)
    elif isinstance(Ybus_, np.ndarray):
        Ybus_ = torch.tensor(Ybus_, dtype=torch.complex64, device=device)

    # Extract components from q_u_delta
    PV_Q = q_u_delta[:, :260]
    PQ_V = q_u_delta[:, 260:1354]
    PQV_theta = q_u_delta[:, 1354:]

    bus_types = torch.tensor(bus_data[:, 1], dtype=torch.int32, device=state.device)
    is_pv = (bus_types == 2) | (bus_types == 3)
    is_pq = (bus_types == 1)
    is_excep_balance = (bus_types == 1) | (bus_types == 2)

    Q_load_values = action[:, num_gens:]
    PQ_V_values = PQ_V

    pv_bus_indices = torch.where(is_pv)[0]
    pq_bus_indices = torch.where(is_pq)[0]
    excep_balance_indices = torch.where(is_excep_balance)[0]

    num_pv_buses = len(pv_bus_indices)
    num_pq_buses = len(pq_bus_indices)

    # Build Vm_combined tensor [batch_size, num_buses]
    Vm_combined = torch.zeros(batch_size, num_buses, dtype=torch.float32, device=state.device)
    pv_Q_load = Q_load_values[:, :num_pv_buses]
    Vm_combined[:, pv_bus_indices] = pv_Q_load
    Vm_combined[:, pq_bus_indices] = PQ_V_values

    # Calculate complex voltage [batch_size, num_buses]
    voltage_tensor = torch.polar(Vm_combined, PQV_theta)

    # Generator calculations
    gen_buses = torch.tensor(gen_data_[:, 0] - 1, dtype=torch.int64, device=state.device)
    Pg_all = action[:, :260]
    Qg_all = PV_Q

    # Create generation power per bus [batch_size, num_buses]
    Pg_per_bus = torch.zeros(batch_size, num_buses, dtype=torch.float32, device=state.device)
    Qg_per_bus = torch.zeros(batch_size, num_buses, dtype=torch.float32, device=state.device)

    for i in range(batch_size):
        Pg_per_bus[i].index_put_([gen_buses], Pg_all[i])
        Qg_per_bus[i].index_put_([gen_buses], Qg_all[i])

    P_load = state[:, :num_buses]
    Q_load = state[:, num_buses:2*num_buses]

    # Calculate node injection power [batch_size, num_buses]
    V_conjugate = torch.conj(voltage_tensor)
    Ybus_expanded = Ybus.unsqueeze(0).expand(batch_size, -1, -1)
    V_conjugate_sum = torch.bmm(Ybus_expanded, voltage_tensor.unsqueeze(-1)).squeeze(-1)
    S_injection = V_conjugate * V_conjugate_sum

    P_injection_ = Pg_per_bus - P_load - torch.real(S_injection)
    Q_injection_ = Qg_per_bus - Q_load + torch.imag(S_injection)

    # Select all nodes except balance node
    P_injection = P_injection_[:, excep_balance_indices]

    # Update Pg for generator 639
    Pg_all_new = Pg_all.clone()
    bus_639_power = (P_load + torch.real(S_injection))[:, 639:640]
    Pg_all_new[:, 639:640] = bus_639_power

    # Select Q_injection for PQ nodes
    Q_injection = Q_injection_[:, pq_bus_indices]
    Qg_all_true = (Q_load - torch.imag(S_injection))[:, pv_bus_indices]

    # Calculate power balance loss
    P_balance = torch.mean(torch.square(P_injection), dim=1)
    Q_balance = torch.mean(torch.square(Q_injection), dim=1)

    # Calculate cost loss
    a = torch.tensor(gencost_data[:, 4], dtype=torch.float32, device=state.device)
    b = torch.tensor(gencost_data[:, 5], dtype=torch.float32, device=state.device)
    c = torch.tensor(gencost_data[:, 6], dtype=torch.float32, device=state.device)

    scaled_Pg = 100 * Pg_all_new
    a_expanded = a.unsqueeze(0).expand(batch_size, -1)
    b_expanded = b.unsqueeze(0).expand(batch_size, -1)
    c_expanded = c.unsqueeze(0).expand(batch_size, -1)

    cost_loss = torch.sum(a_expanded * torch.square(scaled_Pg) + b_expanded * scaled_Pg + c_expanded, dim=1)

    # Calculate reactive power limit violations
    Q_min = torch.tensor(gen_data_[:, 4] / 100, dtype=torch.float32, device=state.device)
    Q_max = torch.tensor(gen_data_[:, 3] / 100, dtype=torch.float32, device=state.device)
    Q_min_expanded = Q_min.unsqueeze(0).expand(batch_size, -1)
    Q_max_expanded = Q_max.unsqueeze(0).expand(batch_size, -1)

    Qg_violations_upper = torch.maximum(Qg_all_true - Q_max_expanded, torch.tensor(0.0, device=state.device))
    Qg_violations_lower = torch.maximum(Q_min_expanded - Qg_all_true, torch.tensor(0.0, device=state.device))
    total_reactive_loss = torch.sum(Qg_violations_upper + Qg_violations_lower, dim=1)

    # Calculate active power limit violations
    P_min = torch.tensor(gen_data_[:, 9] / 100, dtype=torch.float32, device=state.device)
    P_max = torch.tensor(gen_data_[:, 8] / 100, dtype=torch.float32, device=state.device)
    P_min_expanded = P_min.unsqueeze(0).expand(batch_size, -1)
    P_max_expanded = P_max.unsqueeze(0).expand(batch_size, -1)

    Pg_violations_upper = torch.maximum(Pg_all_new - P_max_expanded, torch.tensor(0.0, device=state.device))
    Pg_violations_lower = torch.maximum(P_min_expanded - Pg_all_new, torch.tensor(0.0, device=state.device))
    total_active_loss = torch.sum(Pg_violations_upper + Pg_violations_lower, dim=1) / num_buses

    # Calculate voltage limit violations
    V_min = torch.ones(batch_size, bus_data[:, -1].shape[0], device=state.device) * 0.9
    V_max = torch.ones(batch_size, bus_data[:, -1].shape[0], device=state.device) * 1.1
    V_magnitudes = torch.abs(voltage_tensor)

    V_violations_upper = torch.maximum(V_magnitudes - V_max, torch.tensor(0.0, device=state.device))
    V_violations_lower = torch.maximum(V_min - V_magnitudes, torch.tensor(0.0, device=state.device))
    total_voltage_loss = torch.sum(V_violations_upper + V_violations_lower, dim=1)

    # Calculate line flow limit violations (active branches only)
    active_mask = branch_data[:, 10].astype(int) == 1
    active_branches = branch_data[active_mask]
    arr = active_branches[:, [0, 1, 5, 7]]
    keys = np.array([tuple(sorted(pair)) for pair in arr[:, :2]])
    unique_keys, indices = np.unique(keys, axis=0, return_inverse=True)

    sums = np.zeros(len(unique_keys))
    ratios = np.zeros(len(unique_keys))

    for i in range(len(arr)):
        from_bus, to_bus, rateA, ratio = arr[i]
        ratio = 1.0 if ratio == 0 else ratio
        sums[indices[i]] += rateA
        ratios[indices[i]] = ratio

    result = np.column_stack((unique_keys, sums, ratios/2))

    from_buses = torch.tensor(result[:, 0] - 1, dtype=torch.int64, device=state.device)
    to_buses = torch.tensor(result[:, 1] - 1, dtype=torch.int64, device=state.device)
    line_limits = torch.tensor(result[:, 2] / 100.0, dtype=torch.float32, device=state.device)
    line_limits_expanded = line_limits.unsqueeze(0).expand(batch_size, -1)

    # Calculate line flows
    V_from = voltage_tensor[:, from_buses]
    V_to = voltage_tensor[:, to_buses]
    Y_ij = Ybus_[from_buses, to_buses]
    Y_ij_expanded = Y_ij.unsqueeze(0).expand(batch_size, -1)
    I_ij = Y_ij_expanded * (V_from - V_to)
    flows = torch.conj(V_from) * I_ij
    flow_magnitudes = torch.abs(flows)

    line_violations = torch.maximum(flow_magnitudes - line_limits_expanded, torch.tensor(0.0, device=state.device))
    total_line_loss = torch.sum(line_violations, dim=1)

    # Calculate MSE losses
    mse_q = torch.mean(torch.square(PV_Q - q), dim=1)
    mse_delta = torch.mean(torch.square(PQV_theta - delta), dim=1)
    mse_u = torch.mean(torch.square(PQ_V - u), dim=1)

    # Calculate balance node angle loss
    balance_theta_loss = torch.square(PQV_theta[:, 639] - balance_theta)

    return (
        torch.sqrt(P_balance / num_gens),
        torch.sqrt(Q_balance / num_gens),
        balance_theta_loss,
        cost_loss,
        total_active_loss / num_gens,
        total_reactive_loss / num_gens,
        total_voltage_loss / num_buses,
        total_line_loss / batch_size,
        mse_q,
        mse_delta,
        mse_u
    )


def power_flow_equations_evaluation(case1354, state, action, q_u_delta):
    """
    Power flow equation evaluation for validation (NumPy version).
    
    Args:
        case1354: Power system case data dictionary
        state: State array
        action: Action array
        q_u_delta: Predicted q, u, delta values
    
    Returns:
        tuple: (P_balance, Q_balance) power flow errors
    """
    bus_data = case1354['bus']
    gen_data_ = case1354['gen']
    branch_data = case1354['branch']

    num_buses = bus_data.shape[0]
    num_gens = gen_data_.shape[0]

    Ybus = calculate_ybus(branch_data, num_buses, bus_data).numpy()

    P_balance = 0
    Q_balance = 0

    PV_Q = q_u_delta[:260]
    PQ_V = q_u_delta[260:1354]

    bus_types = case1354['bus'][:, 1]

    Vm_combined = []
    PQ_index = 0
    Q_load_index = 0

    for bus_type in bus_types:
        if bus_type == 2 or bus_type == 3:
            Vm_combined.append(action[num_gens:][Q_load_index])
            Q_load_index += 1
        elif bus_type == 1:
            Vm_combined.append(PQ_V[PQ_index])
            PQ_index += 1

    PQV_theta = q_u_delta[1354:]
    Vm_combined = np.array(Vm_combined)
    voltage_tensor = Vm_combined * np.exp(1j * PQV_theta)

    for i in range(num_buses):
        P_load = state[i]
        Q_load = state[i + num_buses]

        Pg = np.sum(action[:260][gen_data_[:, 0] == (i + 1)])
        Qg = np.sum(PV_Q[gen_data_[:, 0] == (i + 1)])

        V_conjugate_sum = np.dot(Ybus[i, :], voltage_tensor)

        P_injection = Pg - P_load - np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum)
        Q_injection = Qg - Q_load + np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum)

        # if bus_data[i, 1] == 3:
        #     P_injection = 0
        #     Pg = P_load + np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum)
        # else:
        #     P_injection = Pg - P_load - np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum)

        # if bus_data[i, 1] == 1:
        #     Q_injection = Qg - Q_load + np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum)
        # else:
        #     Q_injection = 0
        #     Qg = Q_load - np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum)

        P_balance += (P_injection)**2
        Q_balance += (Q_injection)**2

    return (np.sqrt(P_balance / num_buses), np.sqrt(Q_balance / num_buses))


def AC_optimal_power_flow_equations_evaluation(case1354, state, action, q, u, delta):
    """
    Comprehensive ACOPF evaluation with all constraints.
    
    Args:
        case1354: Power system case data dictionary
        state: State array [P_load, Q_load]
        action: Action array [Pg, V_setpoint]
        q: Reactive power predictions
        u: Voltage magnitude predictions
        delta: Voltage angle predictions
    
    Returns:
        tuple: All power flow and constraint metrics
    """
    bus_data = case1354['bus']
    gen_data_ = case1354['gen']
    branch_data = case1354['branch']
    gencost_data = case1354['gencost']

    num_buses = bus_data.shape[0]
    num_gens = gen_data_.shape[0]
    num_branchs = int(np.sum(branch_data[:, 10].astype(int) == 1))  # active branches only

    Ybus = calculate_ybus(branch_data, num_buses, bus_data).numpy()

    P_balance = 0
    Q_balance = 0
    total_line_loss = 0
    total_voltage_loss = 0
    total_active_loss = 0
    total_reactive_loss = 0
    total_cost_loss = 0

    PV_Q = np.array(q)
    PQ_V = np.array(u)

    bus_types = bus_data[:, 1]

    Vm_combined = []
    PQ_index = 0
    Q_load_index = 0

    for bus_type in bus_types:
        if bus_type == 2 or bus_type == 3:
            Vm_combined.append(action[num_gens:][Q_load_index])
            Q_load_index += 1
        elif bus_type == 1:
            Vm_combined.append(PQ_V[PQ_index])
            PQ_index += 1

    Vm_combined = np.array(Vm_combined)
    PQV_theta = np.array(delta)
    voltage_tensor = Vm_combined * np.exp(1j * PQV_theta)

    for i in range(num_buses):
        P_load = state[i]
        Q_load = state[i + num_buses]

        Pg = np.sum(action[:260][gen_data_[:, 0] == (i + 1)])
        Qg = np.sum(PV_Q[gen_data_[:, 0] == (i + 1)])

        V_conjugate_sum = np.sum(Ybus[i, :] * voltage_tensor)

        P_injection = Pg - P_load - np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum)
        Q_injection = Qg - Q_load + np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum)

        P_balance += P_injection ** 2
        Q_balance += Q_injection ** 2

        # Calculate generator cost loss
        cost_params = gencost_data[gen_data_[:, 0] == (i + 1)]
        if cost_params.shape[0] > 0:
            a = cost_params[:, 4]
            b = cost_params[:, 5]
            c = cost_params[:, 6]
            cost_loss = np.sum(a * (100*action[:260][gen_data_[:, 0] == (i + 1)])**2 + 
                              b * 100 * action[:260][gen_data_[:, 0] == (i + 1)] + c)
            total_cost_loss += cost_loss

        # Calculate reactive power limits
        Q_min = np.sum(gen_data_[gen_data_[:, 0] == (i + 1), 4]) / 100
        Q_max = np.sum(gen_data_[gen_data_[:, 0] == (i + 1), 3]) / 100
        if Qg > Q_max:
            reactive_loss = np.abs(Qg - Q_max)
            total_reactive_loss += reactive_loss
        elif Qg < Q_min:
            reactive_loss = np.abs(Q_min - Qg)
            total_reactive_loss += reactive_loss

        # Calculate active power limits
        P_min = np.sum(gen_data_[gen_data_[:, 0] == (i + 1), 9]) / 100
        P_max = np.sum(gen_data_[gen_data_[:, 0] == (i + 1), 8]) / 100

        # if np.abs(Pg) > 0.0001:
            # print(i," ",Pg)
        if Pg > P_max:
            active_loss = np.abs(Pg - P_max)
            total_active_loss += active_loss
            # print(i," ",Pg," ",P_max)
        elif Pg < P_min:
            active_loss = np.abs(P_min - Pg)
            total_active_loss += active_loss
            # print(i," ",Pg," ",P_min)

        # Calculate voltage limits
        V_min = 0.9
        V_max = 1.1
        V_magnitude = np.abs(voltage_tensor[i])
        if V_magnitude > V_max:
            voltage_loss = np.abs(V_magnitude - V_max)
            total_voltage_loss += voltage_loss
        elif V_magnitude < V_min:
            voltage_loss = np.abs(V_min - V_magnitude)
            total_voltage_loss += voltage_loss

    Ybus_ = calculate_ybus_numpy(branch_data, num_buses, bus_data)
    # Active branches only — avoids inflating line limits with disconnected branch ratings
    active_mask = branch_data[:, 10].astype(int) == 1
    active_branches = branch_data[active_mask]
    arr = active_branches[:, [0, 1, 5, 7]]
    keys = np.array([tuple(sorted(pair)) for pair in arr[:, :2]])
    unique_keys, indices = np.unique(keys, axis=0, return_inverse=True)

    sums = np.zeros(len(unique_keys))
    ratios = np.zeros(len(unique_keys))

    for i in range(len(arr)):
        from_bus, to_bus, rateA, ratio = arr[i]
        ratio = 1.0 if ratio == 0 else ratio
        sums[indices[i]] += rateA
        ratios[indices[i]] = ratio

    result = np.column_stack((unique_keys, sums, ratios/2))

    from_buses = (result[:, 0] - 1).astype(int)
    to_buses = (result[:, 1] - 1).astype(int)
    line_limits = result[:, 2] / 100.0

    V_from = voltage_tensor[from_buses] / result[:, -1]
    V_to = voltage_tensor[to_buses]

    Y_ij = Ybus_[from_buses, to_buses]
    I_ij = Y_ij * (V_from - V_to)
    flows = np.conj(V_from) * I_ij
    flow_magnitudes = np.abs(flows)

    total_line_loss = np.sum(np.maximum(flow_magnitudes - line_limits, 0.0))
    # print(total_active_loss)
    return (
        P_balance / num_buses,
        Q_balance / num_buses,
        total_cost_loss,
        total_active_loss / num_gens,
        total_reactive_loss / num_gens,
        total_voltage_loss / num_buses,
        total_line_loss / num_branchs
    )
