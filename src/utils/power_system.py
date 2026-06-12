"""
Power system calculation functions (Ybus, power flow equations).
"""
import numpy as np
import torch


def calculate_ybus(branch_data, num_buses, bus_data):
    """
    Calculate the bus admittance matrix (Ybus) including shunt admittances.
    Branches with status=0 (column index 10) are skipped.
    
    Args:
        branch_data: Branch data array [from_bus, to_bus, r, x, b, ...]
        num_buses: Number of buses
        bus_data: Bus data array
    
    Returns:
        torch.Tensor: Complex Ybus matrix
    """
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    
    for branch in branch_data:
        if int(branch[10]) == 0:   # disconnected line — skip
            continue
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        resistance = branch[2]
        reactance = branch[3]
        b = branch[4]
        impedance = resistance + 1j * reactance
        Y = 1 / impedance
        
        if branch[-5] == 0:
            ratio = 1.0
            angle_rad = np.deg2rad(branch[-4])
        else:
            ratio = branch[-5]
            angle_rad = np.deg2rad(branch[-4])
        
        Ybus[from_bus, from_bus] += (Y + 1j * (b / 2)) / ratio**2
        Ybus[to_bus, to_bus] += (Y + 1j * (b / 2))
        Ybus[from_bus, to_bus] -= Y * (ratio * np.exp(1j * angle_rad)) / ratio**2
        Ybus[to_bus, from_bus] -= Y * (ratio * np.exp(-1j * angle_rad)) / ratio**2

    # Add shunt admittances
    for i in range(num_buses):
        Gs = bus_data[i][4]
        Bs = bus_data[i][5]
        Ybus[i, i] += Gs / 100 + 1j * Bs / 100
    
    return torch.tensor(Ybus, dtype=torch.complex64)


def calculate_ybus_no_shunt(branch_data, num_buses, bus_data):
    """
    Calculate Ybus without shunt admittances (for line flow calculations).
    Branches with status=0 (column index 10) are skipped.
    
    Args:
        branch_data: Branch data array
        num_buses: Number of buses
        bus_data: Bus data array
    
    Returns:
        torch.Tensor: Complex Ybus matrix without shunt elements
    """
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    
    for branch in branch_data:
        if int(branch[10]) == 0:   # disconnected line — skip
            continue
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        resistance = branch[2]
        reactance = branch[3]
        b = branch[4]
        impedance = resistance + 1j * reactance
        Y = 1 / impedance
        
        if branch[-5] == 0:
            ratio = 1.0
            angle_rad = np.deg2rad(branch[-4])
        else:
            ratio = branch[-5]
            angle_rad = np.deg2rad(branch[-4])
        
        Ybus[from_bus, from_bus] += (Y + 1j * (b / 2)) / ratio**2
        Ybus[to_bus, to_bus] += (Y + 1j * (b / 2))
        Ybus[from_bus, to_bus] -= Y * (ratio * np.exp(1j * angle_rad)) / ratio**2
        Ybus[to_bus, from_bus] -= Y * (ratio * np.exp(-1j * angle_rad)) / ratio**2

    return torch.tensor(Ybus, dtype=torch.complex64)


def calculate_ybus_numpy(branch_data, num_buses, bus_data):
    """
    NumPy version of Ybus calculation for evaluation.
    Branches with status=0 (column index 10) are skipped.
    
    Args:
        branch_data: Branch data array
        num_buses: Number of buses
        bus_data: Bus data array
    
    Returns:
        np.ndarray: Complex Ybus matrix
    """
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    
    for branch in branch_data:
        if int(branch[10]) == 0:   # disconnected line — skip
            continue
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        resistance = branch[2]
        reactance = branch[3]
        b = branch[4]
        impedance = resistance + 1j * reactance
        Y = 1 / impedance
        
        if branch[-5] == 0:
            ratio = 1.0
            angle_rad = np.deg2rad(branch[-4])
        else:
            ratio = branch[-5]
            angle_rad = np.deg2rad(branch[-4])
        
        Ybus[from_bus, from_bus] += (Y + 1j * (b / 2)) / ratio**2
        Ybus[to_bus, to_bus] += (Y + 1j * (b / 2))
        Ybus[from_bus, to_bus] -= Y / ratio * np.exp(1j * angle_rad)
        Ybus[to_bus, from_bus] -= Y / ratio * np.exp(-1j * angle_rad)

    return Ybus
