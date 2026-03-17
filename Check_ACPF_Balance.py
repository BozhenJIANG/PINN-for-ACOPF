import numpy as np
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from case118 import *  
import numpy as np

def calculate_ybus(branch_data, num_buses, bus_data):
    # 计算导纳矩阵 Ybus
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    for _i, branch in enumerate(branch_data):
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        resistance = branch[2]
        reactance = branch[3]
        b = branch[4]  # 提取无功导纳参数
        impedance = resistance + 1j * reactance
        Y = 1 / impedance
        if branch[-5] == 0:
            ratio = 1.0
            angle_rad = np.deg2rad(branch[-4])
        else:
            ratio = branch[-5]
            angle_rad = np.deg2rad(branch[-4])

        Ybus[from_bus, from_bus] += (Y + 1j * (b / 2))/ ratio**2
        Ybus[to_bus, to_bus] += (Y + 1j * (b / 2))
        
        Ybus[from_bus, to_bus] -= Y * (ratio * np.exp(1j * angle_rad))/ ratio**2
        Ybus[to_bus, from_bus] -= Y * (ratio * np.exp(-1j * angle_rad))/ ratio**2

    for i in range(num_buses):
        Gs = bus_data[i][4]  # 对地导纳的实部
        Bs = bus_data[i][5]  # 对地导纳的虚部
        Ybus[i, i] += Gs/100 + 1j * Bs/100 # 更新对角元素       
    
    return Ybus

def calculate_ybus_(branch_data, num_buses, bus_data):
    # 计算导纳矩阵 Ybus
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    for _i, branch in enumerate(branch_data):
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        resistance = branch[2]
        reactance = branch[3]
        b = branch[4]  # 提取无功导纳参数
        impedance = resistance + 1j * reactance
        Y = 1 / impedance
        if branch[-5] == 0:
            ratio = 1.0
            angle_rad = np.deg2rad(branch[-4])
        else:
            ratio = branch[-5]
            angle_rad = np.deg2rad(branch[-4])
            
        Ybus[from_bus, from_bus] += (Y + 1j * (b / 2))/ ratio**2
        Ybus[to_bus, to_bus] += (Y + 1j * (b / 2))
        
        Ybus[from_bus, to_bus] -= Y * (ratio * np.exp(1j * angle_rad))/ ratio**2
        Ybus[to_bus, from_bus] -= Y * (ratio * np.exp(-1j * angle_rad))/ ratio**2

    return Ybus

def power_flow_equations_evaluation(case118, state, action, q_u_delta):  
    # Extract necessary data  
    bus_data = case118['bus']  
    gen_data_ = case118['gen']  
    branch_data = case118['branch']  
    gencost_data = case118['gencost']  
  
    # Number of buses  
    num_buses = bus_data.shape[0]  
    num_gens = gen_data_.shape[0]  
    num_branchs = branch_data.shape[0]  
  
    # Calculate Ybus  
    Ybus = calculate_ybus(branch_data, num_buses, bus_data)
  
    # Initialize power balance equations  
    P_balance = 0  
    Q_balance = 0  
  
    PV_Q = q_u_delta[:54]  
    PQ_V = q_u_delta[54:118]  
    # print(q_u_delta)
  
    # Bus type information (1, 2, 3)  
    bus_types = case118['bus'][:, 1]  
  
    # Initialize an empty Vm list  
    Vm_combined = []  
    PQ_index = 0  
    Q_load_index = 0  


    # Initialize power balance equations
    P_balance = 0
    Q_balance = 0
    total_line_loss = 0
    total_voltage_loss = 0
    total_active_loss = 0
    total_reactive_loss = 0
    total_cost_loss = 0
  
    # Fill Vm_combined according to bus types  
    for _i_bus, bus_type in enumerate(bus_types):  

        if bus_type == 2 or bus_type == 3:  
            Vm_combined.append(action[num_gens:][Q_load_index])  
            Q_load_index += 1  
        elif bus_type == 1:  
            # Value from PQ_V  
            Vm_combined.append(PQ_V[PQ_index])  
            PQ_index += 1 
            
    PQV_theta = q_u_delta[118:]  
  
    Vm_combined = np.array(Vm_combined)  
    # print(PV_Q)
    # print(Vm_combined)
    voltage_tensor = Vm_combined * np.exp(1j * PQV_theta)  # Voltage at bus 

    # Real and reactive power injections  
    for i in range(num_buses):  
        # Power demand at bus  
        P_load = state[i]  # Load active power  
        Q_load = state[i + num_buses]  # Load reactive power  
  
        # Generator power outputs  
        Pg = np.sum(action[:54][gen_data_[:, 0] == (i + 1)])  # Active power from generators  
        Qg = np.sum(PV_Q[gen_data_[:, 0] == (i + 1)])  # Reactive power from generators  
  
        # Power balance equations  
        V_conjugate_sum = np.dot(Ybus[i, :], voltage_tensor)  
          
        P_injection = Pg - P_load - np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum)  
        Q_injection = Qg - Q_load + np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum) 

        P_balance += P_injection ** 2
        Q_balance += Q_injection ** 2

        # Calculate generator cost loss
        cost_params = gencost_data[gen_data_[:, 0] == (i + 1)]
        if cost_params.shape[0] > 0:
            a = cost_params[:, 4]  # 二次成本系数
            b = cost_params[:, 5]  # 一次成本系数
            c = cost_params[:, 6]  # 固定成本
            cost_loss = np.sum(a * (100*action[:54][gen_data_[:, 0] == (i + 1)])**2 + b * 100 * action[:54][gen_data_[:, 0] == (i + 1)] + c)
            total_cost_loss += cost_loss

        # Calculate reactive power limits
        Q_min = np.sum(gen_data_[gen_data_[:, 0] == (i + 1), 4]) / 100  # Minimum reactive output
        Q_max = np.sum(gen_data_[gen_data_[:, 0] == (i + 1), 3]) / 100  # Maximum reactive output
        if Qg > Q_max:
            reactive_loss = np.abs(Qg - Q_max)  # L1 norm
            total_reactive_loss += reactive_loss
        elif Qg < Q_min:
            reactive_loss = np.abs(Q_min - Qg)  # L1 norm
            total_reactive_loss += reactive_loss

        # Calculate active power limits
        P_min = np.sum(gen_data_[gen_data_[:, 0] == (i + 1), 9]) / 100  # Minimum active output
        P_max = np.sum(gen_data_[gen_data_[:, 0] == (i + 1), 8]) / 100  # Maximum active output
        if Pg > P_max:
            active_loss = np.abs(Pg - P_max)  # L1 norm
            total_active_loss += active_loss
        elif Pg < P_min:
            active_loss = np.abs(P_min - Pg)  # L1 norm
            total_active_loss += active_loss

        # Calculate voltage limits
        V_min = 0.94 # Minimum voltage limit
        V_max = 1.06  # Maximum voltage limit
        
        V_magnitude = np.abs(voltage_tensor[i])  # Voltage magnitude
        if V_magnitude > V_max:
            voltage_loss = np.abs(V_magnitude - V_max)  # L1 norm
            total_voltage_loss += voltage_loss
        elif V_magnitude < V_min:
            voltage_loss = np.abs(V_min - V_magnitude)  # L1 norm
            total_voltage_loss += voltage_loss

    Ybus_ = calculate_ybus_(branch_data, num_buses, bus_data)
    # Calculate line limit losses
    for branch in branch_data:
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        line_limit = branch[5] / 100.0  # Line maximum flow limit

        # Calculate voltages at buses
        V_from = voltage_tensor[from_bus]
        V_to = voltage_tensor[to_bus]
        
        # 计算支路电流
        Y_ij = Ybus_[from_bus, to_bus]
        I_ij = Y_ij * (V_from - V_to)
        
        # 计算支路潮流
        flow = V_from * np.conj(I_ij)

        # Calculate actual flow magnitude
        flow_magnitude = np.abs(flow)

        # Calculate loss if flow exceeds line limit
        if flow_magnitude > line_limit:
            line_loss = np.abs(flow_magnitude - line_limit)  # L1 norm
            total_line_loss += line_loss

    return (np.sqrt(P_balance / num_buses), np.sqrt(Q_balance / num_buses), 
            total_cost_loss, total_active_loss / num_gens, 
            total_reactive_loss / num_gens, total_voltage_loss / num_buses, 
            total_line_loss / num_branchs)

case118 = case118()
X_con_test_ = []
with open("./Dataset/X_con_118_test.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_con_test_.append([float(_i) for _i in _data])

X_in_test_ = []
with open("./Dataset/X_in_118_test.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_in_test_.append([float(_i) for _i in _data])

X_other_information_test_ = []
with open("./Dataset/X_other_information_118_test.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_other_information_test_.append([float(_i) for _i in _data])



X_in_test = np.array(X_in_test_)
X_in_test[:,:54] = X_in_test[:,:54]/100
X_con_test = np.array(X_con_test_)/100
X_other_information_test = np.array(X_other_information_test_)

base_p_error = 0
base_q_error = 0
base_cost_error = 0
base_active_error = 0
base_reactive_error = 0
base_voltage_error = 0
base_line_error = 0

num_buses = case118['bus'].shape[0]

for i in range(X_con_test.shape[0]):
    _temp_p, _temp_q, _temp_cost, _temp_active, _temp_reactive, _temp_voltage, _temp_line = power_flow_equations_evaluation(case118, X_con_test[i,:], X_in_test[i,:], X_other_information_test[i,:])
    base_p_error += _temp_p
    base_q_error += _temp_q
    base_cost_error += _temp_cost
    base_active_error += _temp_active
    base_reactive_error += _temp_reactive
    base_voltage_error += _temp_voltage
    base_line_error += _temp_line
    # break

_temp_shape = X_con_test.shape[0]

print("BASE error: ",base_p_error/_temp_shape," ",base_q_error/_temp_shape," ",base_cost_error/_temp_shape," ",base_active_error/_temp_shape," ",base_reactive_error/_temp_shape," ",base_voltage_error/_temp_shape," ",base_line_error/_temp_shape)

