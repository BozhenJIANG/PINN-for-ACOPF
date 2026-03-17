import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import traceback
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from case118 import *

torch.cuda.set_device(1)  # 强制使用GPU 1
device = torch.device('cuda:1')
print(f"使用设备: {device}")

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./acopf_train_log.txt',
                    filemode='w')

logging.info('Start')

def beta_silu(x, beta=1.0):
    return x * torch.sigmoid(beta * x)

def min_max_sigmoid(x):
    return 0.94 + 0.12 * torch.sigmoid(x)

def calculate_ybus(branch_data, num_buses, bus_data):
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    for _i, branch in enumerate(branch_data):
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

    for i in range(num_buses):
        Gs = bus_data[i][4]
        Bs = bus_data[i][5]
        Ybus[i, i] += Gs/100 + 1j * Bs / 100
    
    return torch.tensor(Ybus, dtype=torch.complex64)

def calculate_ybus_(branch_data, num_buses, bus_data):
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    for _i, branch in enumerate(branch_data):
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

def power_flow_equations(case118, state, action, q, u, delta, q_u_delta, balance_theta):
    bus_data = case118['bus']
    gen_data_ = case118['gen']
    branch_data = case118['branch']
    gencost_data = case118['gencost']

    num_buses = bus_data.shape[0]
    num_gens = gen_data_.shape[0]

    Ybus = calculate_ybus(branch_data, num_buses, bus_data)
    Ybus_ = calculate_ybus_(branch_data, num_buses, bus_data)

    PV_Q = q_u_delta[:54]
    PQ_V = q_u_delta[54:118]
    PQV_theta = q_u_delta[118:]

    bus_types = torch.tensor(bus_data[:, 1], dtype=torch.int32)
    
    is_pv = (bus_types == 2) | (bus_types == 3)
    is_pq = (bus_types == 1)
    is_excep_balance = (bus_types == 1) | (bus_types == 2)

    Q_load_values = action[num_gens:]
    PQ_V_values = PQ_V
    
    pv_bus_indices = torch.where(is_pv)[0]
    pq_bus_indices = torch.where(is_pq)[0]
    excep_balance_indices = torch.where(is_excep_balance)[0]

    Vm_combined = torch.zeros(num_buses, dtype=torch.float32)
    
    Vm_combined = Vm_combined.index_put([pv_bus_indices], 
                                       Q_load_values[torch.arange(len(pv_bus_indices))])
    Vm_combined = Vm_combined.index_put([pq_bus_indices], PQ_V_values)
    
    voltage_tensor = torch.complex(Vm_combined, torch.zeros_like(Vm_combined)) * torch.exp(torch.complex(torch.tensor(0.0), PQV_theta))

    gen_buses = torch.tensor(gen_data_[:, 0] - 1, dtype=torch.int64)
    Pg_all = action[:54]
    Qg_all = PV_Q

    Pg_per_bus = torch.zeros(num_buses, dtype=torch.float32)
    Pg_per_bus = Pg_per_bus.index_put([gen_buses], Pg_all)
    
    Qg_per_bus = torch.zeros(num_buses, dtype=torch.float32)
    Qg_per_bus = Qg_per_bus.index_put([gen_buses], Qg_all)
    
    P_load = state[:num_buses]
    Q_load = state[num_buses:2*num_buses]

    V_conjugate = torch.conj(voltage_tensor)
    V_conjugate_sum = torch.matmul(Ybus, voltage_tensor)
    S_injection = V_conjugate * V_conjugate_sum
    
    P_injection_ = Pg_per_bus - P_load - torch.real(S_injection)
    Q_injection_ = Qg_per_bus - Q_load + torch.imag(S_injection)

    P_injection = P_injection_[excep_balance_indices]

    Pg_all_new = Pg_all.clone()
    Pg_all_new[29] = (P_load + torch.real(S_injection))[29]

    Q_injection = Q_injection_[pq_bus_indices]
    Qg_all_true = (Q_load - torch.imag(S_injection))[pv_bus_indices]

    P_balance = torch.sum(torch.square(P_injection))
    Q_balance = torch.sum(torch.square(Q_injection))

    a = torch.tensor(gencost_data[:, 4], dtype=torch.float32)
    b = torch.tensor(gencost_data[:, 5], dtype=torch.float32)
    c = torch.tensor(gencost_data[:, 6], dtype=torch.float32)
    
    scaled_Pg = 100 * Pg_all_new
    cost_loss = torch.sum(a * torch.square(scaled_Pg) + b * scaled_Pg + c)

    Q_min = torch.tensor(gen_data_[:, 4] / 100, dtype=torch.float32)
    Q_max = torch.tensor(gen_data_[:, 3] / 100, dtype=torch.float32)
    
    Qg_violations_upper = torch.maximum(Qg_all_true - Q_max, torch.tensor(0.0))
    Qg_violations_lower = torch.maximum(Q_min - Qg_all_true, torch.tensor(0.0))
    total_reactive_loss = torch.sum(Qg_violations_upper + Qg_violations_lower)

    P_min = torch.tensor(gen_data_[:, 9] / 100, dtype=torch.float32)
    P_max = torch.tensor(gen_data_[:, 8] / 100, dtype=torch.float32)
    
    Pg_violations_upper = torch.maximum(Pg_all_new - P_max, torch.tensor(0.0))
    Pg_violations_lower = torch.maximum(P_min - Pg_all_new, torch.tensor(0.0))
    total_active_loss = torch.sum(Pg_violations_upper + Pg_violations_lower)

    V_min = torch.ones(bus_data[:, -1].shape) * 0.94
    V_max = torch.ones(bus_data[:, -1].shape) * 1.06
        
    V_magnitudes = torch.abs(voltage_tensor)

    V_violations_upper = torch.maximum(V_magnitudes - V_max, torch.tensor(0.0))
    V_violations_lower = torch.maximum(V_min - V_magnitudes, torch.tensor(0.0))
    total_voltage_loss = torch.sum(V_violations_upper + V_violations_lower)

    arr = branch_data[:, [0, 1, 5, 7]]
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

    from_buses = torch.tensor(result[:, 0] - 1, dtype=torch.int64)
    to_buses = torch.tensor(result[:, 1] - 1, dtype=torch.int64)
    line_limits = torch.tensor(result[:, 2] / 100.0, dtype=torch.float32)

    V_from = torch.index_select(voltage_tensor, 0, from_buses) / torch.tensor(result[:, -1], dtype=torch.complex64)
    V_to = torch.index_select(voltage_tensor, 0, to_buses)
    Y_ij = Ybus_[from_buses, to_buses]
    I_ij = Y_ij * (V_from - V_to)
    flows = torch.conj(V_from) * I_ij
    flow_magnitudes = torch.abs(flows)
    
    line_violations = torch.maximum(flow_magnitudes - line_limits, torch.tensor(0.0))
    total_line_loss = torch.sum(line_violations)

    mse_q = torch.mean(torch.square(PV_Q - q))
    mse_delta = torch.mean(torch.square(PQV_theta - delta))
    mse_u = torch.mean(torch.square(PQ_V - u))

    balance_theta_loss = torch.square(PQV_theta[68] - balance_theta)

    return (P_balance, 
            Q_balance, 
            balance_theta_loss,
            cost_loss, 
            total_active_loss, 
            total_reactive_loss, 
            total_voltage_loss, 
            total_line_loss, 
            mse_q, 
            mse_delta, 
            mse_u)

class BetaSiLU(nn.Module):
    def __init__(self, beta=1.0):
        super(BetaSiLU, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class MinMaxSigmoid(nn.Module):
    def forward(self, x):
        return 0.94 + 0.12 * torch.sigmoid(x)

class ACOPFM(nn.Module):
    def __init__(self, act_dim, intermediate_dim, state_dim, latent_dim, limits, model_type="FC", l1_reg=0.01):
        super(ACOPFM, self).__init__()
        self.act_dim = act_dim
        self.intermediate_dim = intermediate_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.model_type = model_type
        self.l1_reg = l1_reg
        self.limits = limits
        self.limits_q = [(_temp_data[4]/100, _temp_data[3]/100) for _temp_data in case118["gen"]]
                
        self.encoder = self.encoder_model()
        self.pinn_pf_ = self.pinn_pf_model()

    def pinn_pf_model(self):
        if self.model_type == "FC":
            return PINN_PF_Model(self.act_dim, self.state_dim, self.intermediate_dim, self.limits_q)

    def pinn_pf(self, x, y):
        return self.pinn_pf_(x, y)

    def encoder_model(self):
        if self.model_type == "FC":
            return EncoderModel(self.state_dim, self.act_dim, self.intermediate_dim, self.limits)
    
    def encode(self, y):
        return self.encoder(y)

class PINN_PF_Model(nn.Module):
    def __init__(self, act_dim, state_dim, intermediate_dim, limits_q):
        super(PINN_PF_Model, self).__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.intermediate_dim = intermediate_dim
        self.limits_q = limits_q
        
        # Action branch
        self.x_branch = nn.Sequential(
            nn.Linear(act_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, 2 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(2 * intermediate_dim, 4 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(4 * intermediate_dim, 8 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(8 * intermediate_dim, 16 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(16 * intermediate_dim, 32 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(32 * intermediate_dim, 8 * intermediate_dim),
            nn.SiLU()
        )
        
        # State branch
        self.y_branch = nn.Sequential(
            nn.Linear(state_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, 2 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(2 * intermediate_dim, 4 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(4 * intermediate_dim, 8 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(8 * intermediate_dim, 16 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(16 * intermediate_dim, 32 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(32 * intermediate_dim, 8 * intermediate_dim),
            nn.SiLU()
        )
        
        # Combined network
        self.combined_net = nn.Sequential(
            nn.Linear(16 * intermediate_dim, 16 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(16 * intermediate_dim, 32 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(32 * intermediate_dim, 64 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(64 * intermediate_dim, 32 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(32 * intermediate_dim, 236)
        )
        
        self.beta_silu = BetaSiLU(beta=1.0)
        self.min_max_sigmoid = MinMaxSigmoid()

    def forward(self, x, y):
        x_feat = self.x_branch(x)
        y_feat = self.y_branch(y)
        
        combined = torch.cat([x_feat, y_feat], dim=1)
        p7 = self.combined_net(combined)
        
        # Process outputs
        p7_0 = torch.sigmoid(2*p7[:, :54]) #缩放，使其训练更加稳定
        p8_1 = torch.stack([a + (b - a) * p7_0[:, i] for i, (a, b) in enumerate(self.limits_q)], dim=1)
        
        p8_2 = self.min_max_sigmoid(p7[:, 54:118])
        p8_3 = self.beta_silu(p7[:, 118:])
        
        p8 = torch.cat([p8_1, p8_2, p8_3], dim=1)
        return p8

class EncoderModel(nn.Module):
    def __init__(self, state_dim, act_dim, intermediate_dim, limits):
        super(EncoderModel, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.intermediate_dim = intermediate_dim
        self.limits = limits
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, intermediate_dim),
            nn.Linear(intermediate_dim, 2 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(2 * intermediate_dim, 4 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(4 * intermediate_dim, 8 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(8 * intermediate_dim, 16 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(16 * intermediate_dim, 8 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(8 * intermediate_dim, 4 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(4 * intermediate_dim, 2 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(2 * intermediate_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, act_dim),
            nn.Sigmoid()
        )

    def forward(self, y):
        x_0 = self.net(y)
        # Map to action ranges
        x = torch.stack([a + (b - a) * x_0[:, i] for i, (a, b) in enumerate(self.limits)], dim=1)
        return x

case118 = case118()

X_con_train_ = []
with open("./Dataset/X_con_118_train.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_con_train_.append([float(_i) for _i in _data])

X_in_train_ = []
with open("./Dataset/X_in_118_train.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_in_train_.append([float(_i) for _i in _data])

X_other_information_train_ = []
with open("./Dataset/X_other_information_118_train.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_other_information_train_.append([float(_i) for _i in _data])

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

# 转换为PyTorch张量
X_in_train = torch.tensor(X_in_train_, dtype=torch.float32)
X_in_train[:, :54] = X_in_train[:, :54] / 100
X_con_train = torch.tensor(X_con_train_, dtype=torch.float32) / 100
X_other_information_train = torch.tensor(X_other_information_train_, dtype=torch.float32)

X_in_test = torch.tensor(X_in_test_, dtype=torch.float32)
X_in_test[:, :54] = X_in_test[:, :54] / 100
X_con_test = torch.tensor(X_con_test_, dtype=torch.float32) / 100
X_other_information_test = torch.tensor(X_other_information_test_, dtype=torch.float32)

selected_indices = np.random.choice(len(X_con_test), 5000, replace=False)
all_indices = np.arange(len(X_con_test))
unselected_indices = np.setdiff1d(all_indices, selected_indices)

X_in_train_NL = X_in_test[selected_indices,:]
X_con_train_NL = X_con_test[selected_indices,:]
X_other_information_train_NL = X_other_information_test[selected_indices,:]

X_in_test = X_in_test[unselected_indices,:]
X_con_test = X_con_test[unselected_indices,:]
X_other_information_test = X_other_information_test[unselected_indices,:]

state_dim = X_con_test.shape[1]
act_dim = X_in_test.shape[1]

# 训练配置
batch_size = 512
sample_num_for_validate = 50
epochs = 1000
pre_train_epoch = 500

_temp_data = case118["gen"]
limits = [(_temp_data[i][9]/100, _temp_data[i][8]/100) for i in range(len(_temp_data))] + \
         [(0.94, 1.06) for _temp_data in case118["bus"] if _temp_data[1] in [2, 3]]

# 创建数据加载器
train_dataset = TensorDataset(X_in_train, X_con_train, X_other_information_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_dataset_NL = TensorDataset(X_in_train_NL, X_con_train_NL, X_other_information_train_NL)
train_loader_NL = DataLoader(train_dataset_NL, batch_size=batch_size, shuffle=False)


# 训练循环
logging.info('ACOPF Start')

def update_learning_rate(optimizer, epoch, total_epoch=1000, mim_lear=1e-9):
    initial_learning_rate = 1e-3
    
    if epoch < 50:
        new_lr = initial_learning_rate * 1.0
    elif epoch < 80:
        new_lr = initial_learning_rate * 0.5
    elif epoch < 160:
        new_lr = initial_learning_rate * 0.05
    elif epoch < 200:
        new_lr = initial_learning_rate * 0.01
    elif epoch < 500:
        new_lr = initial_learning_rate * 0.005
    else:
        # 线性衰减从 0.005 * initial_learning_rate 到 mim_lear
        start_lr = initial_learning_rate * 0.01
        # 计算衰减比例 (epoch-500) / (total_epoch-500)
        progress = min(1.0, (epoch - 500) / (total_epoch - 500))
        new_lr = start_lr + (mim_lear - start_lr) * progress
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print(f"Learning rate updated to: {new_lr}")
    return new_lr


# 训练函数
def train_step_0(model, x, y, other_variable, optimizer, epoch, penalty_coefficient=0.1):
    optimizer.zero_grad()
    losses = compute_loss(model, x, y, other_variable, epoch, penalty_coefficient)
    
    # 训练encoder
    loss_encoder = losses[0]
    loss_encoder.backward(retain_graph=True)
    optimizer.step()
    
    # 训练pinn_pf
    optimizer.zero_grad()
    loss_pf = losses[1]
    loss_pf.backward()
    optimizer.step()
    
    return losses

# 其他训练步骤函数
def train_step_1(model, x, y, other_variable, optimizer, epoch, penalty_coefficient=0.1):
    """训练PINN_PF模型"""
    optimizer.zero_grad()
    losses = compute_loss(model, x, y, other_variable, epoch, penalty_coefficient)
    
    # 只训练pinn_pf，不训练encoder
    loss = (0 * losses[0] + 1 * losses[1] + 1 * losses[2] + 1 * losses[3] + 
            0 * losses[4] + 0 * losses[5] + 0 * losses[6] + 0 * losses[7] + 
            0 * losses[8] + 0 * losses[9] + 0 * losses[10] + 0 * losses[11])
    
    loss.backward()
    optimizer.step()
    return losses

def train_step_2(model, x, y, other_variable, optimizer, epoch, penalty_coefficient=0.1):
    """只训练encoder的约束损失"""
    optimizer.zero_grad()
    losses = compute_loss(model, x, y, other_variable, epoch, penalty_coefficient)
    
    loss = (0 * losses[0] + 0 * losses[1] + 0 * losses[2] + 0 * losses[3] + 
            1e-9 * losses[4] + 1 * losses[5] + 1 * losses[6] + 1 * losses[7] + 
            1 * losses[8] + 0 * losses[9] + 0 * losses[10] + 0 * losses[11])
    
    loss.backward()
    optimizer.step()
    return losses

def compute_loss(model, x, y, other_variable, epoch, penalty_coefficient):
    x_ = model.encode(y)
    xent_loss = torch.mean((x_ - x)**2)

    if epoch < pre_train_epoch:
        q_u_delta = model.pinn_pf(x, y)
        q_u_delta_loss = torch.mean((q_u_delta - other_variable)**2)
        return (xent_loss, q_u_delta_loss)
    else:
        q_u_delta = model.pinn_pf(x_, y)
        
        p_loss = 0
        q_loss = 0
        theta_balance_loss = 0
        cost_loss = 0
        active_loss = 0
        reactive_loss = 0
        voltage_loss = 0
        line_loss = 0
        pf_q_loss = 0
        pf_delta_loss = 0 
        pf_u_loss = 0
        
        samples = x_.shape[0]
        
        for _i in range(samples):
            q, u, delta = other_variable[_i, :54], other_variable[_i, 54:118], other_variable[_i, 118:]
            balance_theta = other_variable[_i, 118:][68]
            
            (grad_P_balance, grad_Q_balance, grad_theta_balance, grad_cost_loss, 
             grad_active_loss, grad_reactive_loss, grad_voltage_loss, grad_line_loss, 
             grad_mse_q, grad_mse_delta, grad_mse_u) = power_flow_equations(
                 case118, y[_i, :], x_[_i, :], q, u, delta, q_u_delta[_i, :], balance_theta)
            
            p_loss += grad_P_balance
            q_loss += grad_Q_balance
            theta_balance_loss += grad_theta_balance
            cost_loss += grad_cost_loss
            active_loss += grad_active_loss
            reactive_loss += grad_reactive_loss
            voltage_loss += grad_voltage_loss
            line_loss += grad_line_loss
            pf_q_loss += grad_mse_q
            pf_delta_loss += grad_mse_delta
            pf_u_loss += grad_mse_u

        return (penalty_coefficient * xent_loss,
                penalty_coefficient * p_loss / samples,
                penalty_coefficient * q_loss / samples,
                theta_balance_loss / samples,
                penalty_coefficient * cost_loss / samples,
                penalty_coefficient * active_loss,
                penalty_coefficient * reactive_loss,
                penalty_coefficient * voltage_loss,
                penalty_coefficient * line_loss / samples,
                penalty_coefficient * pf_q_loss / samples,
                penalty_coefficient * pf_delta_loss / samples,
                penalty_coefficient * pf_u_loss / samples)

# 验证函数
def validate_model(model, case118, X_con_test, sample_num_for_validate=50):
    """验证模型性能"""
    model.eval()
    pinn_all_p_error = 0
    pinn_all_q_error = 0
    
    with torch.no_grad():
        # 随机选择样本进行验证
        _temp_index = np.random.choice(len(X_con_test), sample_num_for_validate, replace=False)
        
        for i in _temp_index:
            # 使用encoder预测动作
            X_in_pre = model.encode(X_con_test[i:i+1])
            _pre_data_pinn = model.pinn_pf(X_in_pre, X_con_test[i:i+1])
            
            # 转换为numpy进行计算
            state_np = X_con_test[i].numpy()
            action_np = X_in_pre[0].numpy()
            q_u_delta_np = _pre_data_pinn[0].numpy()
            
            # 计算功率流误差
            _temp_p, _temp_q = power_flow_equations_evaluation(
                case118, state_np, action_np, q_u_delta_np)
            
            pinn_all_p_error += np.sqrt(_temp_p)
            pinn_all_q_error += np.sqrt(_temp_q)
    
    return pinn_all_p_error / sample_num_for_validate, pinn_all_q_error / sample_num_for_validate

# AC-PF评估函数
def power_flow_equations_evaluation(case118, state, action, q_u_delta):
    """功率流方程评估 - 保持原有的numpy实现"""
    bus_data = case118['bus']
    gen_data_ = case118['gen']
    branch_data = case118['branch']
    gencost_data = case118['gencost']
  
    num_buses = bus_data.shape[0]  
    num_gens = gen_data_.shape[0]  
  
    # Calculate Ybus  
    Ybus = calculate_ybus(branch_data, num_buses, bus_data).numpy()
  
    P_balance = 0  
    Q_balance = 0  
  
    PV_Q = q_u_delta[:54]
    PQ_V = q_u_delta[54:118]

    bus_types = case118['bus'][:, 1]  
  
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

    PQV_theta = q_u_delta[118:]  
    Vm_combined = np.array(Vm_combined)  
    voltage_tensor = Vm_combined * np.exp(1j * PQV_theta)
  
    for i in range(num_buses):  
        P_load = state[i]  
        Q_load = state[i + num_buses]  
  
        Pg = np.sum(action[:54][gen_data_[:, 0] == (i + 1)])  
        Qg = np.sum(PV_Q[gen_data_[:, 0] == (i + 1)])  
  
        V_conjugate_sum = np.dot(Ybus[i, :], voltage_tensor)  
          
        P_injection = Pg - P_load - np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum)  
        Q_injection = Qg - Q_load + np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum) 

        if bus_data[i, 1] == 3:
            P_injection = 0
            Pg = P_load + np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum) 
        else:
            P_injection = Pg - P_load - np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum)

        if bus_data[i, 1] == 1:
            Q_injection = Qg - Q_load + np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum) 
        else:
            Q_injection = 0
            Qg = Q_load - np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum) 

        P_balance += (P_injection)**2  
        Q_balance += (Q_injection)**2
            
    return (P_balance / num_buses, Q_balance / num_buses)

# 主训练循环
CFDP_loss = []
best_p_error = float('inf')

# 初始化模型和优化器
model = ACOPFM(act_dim, 64, state_dim, 2, limits, 'FC')
optimizer_1 = optim.Adam([
    {'params': model.encoder.parameters()},
    {'params': model.pinn_pf_.parameters()}
], lr=5e-5, betas=(0.8, 0.9))

# 训练循环 - 简洁版本
logging.info('ACOPF Start')

tqdm_e = tqdm(range(1, epochs + 1), desc='Training', leave=True, unit="epochs")
CFDP_loss = []
best_p_error = float('inf')

for epoch in tqdm_e:
    model.train()
    total_loss = 0
    
    # 更新学习率
    new_lr = update_learning_rate(optimizer_1, epoch)
    
    # 调整惩罚系数
    penalty_coefficient = 1
    
    # 训练步骤
    batch_count = 0
    if epoch < pre_train_epoch:
        # 预训练阶段
        for batch_idx, (train_x, train_y, other_variable) in enumerate(train_loader):
            if batch_idx >= 1:
                losses = train_step_0(model, train_x, train_y, other_variable, optimizer_1, epoch, penalty_coefficient)
                total_loss += losses[0].item()
                batch_count = 1
                break
    else:
        # 正式训练阶段
        # 步骤1.1: 训练PINN_PF
        for batch_idx, (train_x, train_y, other_variable) in enumerate(train_loader_NL):
            if batch_idx >= 8:
                break
            losses = train_step_1(model, train_x, train_y, other_variable, optimizer_1, epoch, penalty_coefficient)
            total_loss += losses[0].item()
            batch_count += 1
        
        # 步骤2: 训练encoder的约束
        for batch_idx, (train_x, train_y, other_variable) in enumerate(train_loader_NL):
            if batch_idx >= 1:
                losses = train_step_2(model, train_x, train_y, other_variable, optimizer_1, epoch, penalty_coefficient)
                total_loss += losses[0].item()
                batch_count += 1
                break
    
    # 验证
    if epoch % 10 == 0 or epoch == 1:
        p_error, q_error = validate_model(model, case118, X_con_test, sample_num_for_validate)
        
        # 记录日志
        logging.info(f'Epoch {epoch} : ACOPF error: {p_error:.3f}, {q_error:.3f}')
        
        # 保存最佳模型
        if p_error < best_p_error:
            best_p_error = p_error
            best_q_error = q_error
            torch.save(model.encoder.state_dict(), "best_ACOPF_encoder.pth")
            torch.save(model.pinn_pf_.state_dict(), "best_ACOPF_pinn_pf.pth")
        
        # 更新进度条显示验证信息
        avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
        tqdm_e.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f} | P-Err: {p_error:.4f} | Q-Err: {q_error:.4f}")
        tqdm_e.set_postfix({'LR': f'{new_lr:.2e}', 'Best P': f'{best_p_error:.4f}'})
    else:
        # 更新进度条显示训练信息
        avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
        tqdm_e.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        tqdm_e.set_postfix({'LR': f'{new_lr:.2e}'})
    
    CFDP_loss.append(avg_loss)

print(f"\n Training completed!")
logging.info(f'ACOPF End - Best P error: {best_p_error:.6f} | Best Q error: {best_q_error:.6f}')

model.encoder.load_state_dict(torch.load('./best_ACOPF_encoder.pth'))
model.pinn_pf_.load_state_dict(torch.load('./best_ACOPF_pinn_pf.pth'))
p_error, q_error = validate_model(model, case118, X_con_test, sample_num_for_validate)
print(f"P-Err: {p_error:.4f} | Q-Err: {q_error:.4f}")

# 使用训练好的模型进行预测
model.eval()
with torch.no_grad():
    X_pre = model.encode(X_con_test)

print(f"ACOPF模型RMSE: {torch.sqrt(torch.mean((X_pre - X_in_test)**2)).item():.6f}")

def calculate_ybus_numpy(branch_data, num_buses, bus_data):
    # Initialize the admittance matrix Ybus
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    
    # Calculate branch admittances
    for branch in branch_data:
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        resistance = branch[2]
        reactance = branch[3]
        b = branch[4]  # Reactive admittance parameter
        impedance = resistance + 1j * reactance
        Y = 1 / impedance
        
        # Ratio and angle handling
        if branch[-5] == 0:
            ratio = 1.0
            angle_rad = np.deg2rad(branch[-4])
        else:
            ratio = branch[-5]
            angle_rad = np.deg2rad(branch[-4])
        
        # Update the admittance matrix
        Ybus[from_bus, from_bus] += (Y + 1j * (b / 2)) / ratio**2
        Ybus[to_bus, to_bus] += (Y + 1j * (b / 2))
        
        Ybus[from_bus, to_bus] -= Y / ratio * np.exp(1j * angle_rad)
        Ybus[to_bus, from_bus] -= Y / ratio * np.exp(1j * angle_rad)

    return Ybus

def AC_optimal_power_flow_equations_evaluation(case118, state, action, q, u, delta, flag = True):
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
    Ybus = (calculate_ybus(branch_data, num_buses, bus_data)).numpy()

    # Initialize power balance equations
    P_balance = 0
    Q_balance = 0
    total_line_loss = 0
    total_voltage_loss = 0
    total_active_loss = 0
    total_reactive_loss = 0
    total_cost_loss = 0

    PV_Q = np.array(q)
    PQ_V = np.array(u)

    # Bus types
    bus_types = bus_data[:, 1]

    # Initialize Vm_combined
    Vm_combined = []
    PQ_index = 0
    Q_load_index = 0

    # Fill Vm_combined based on bus types
    for bus_type in bus_types:
        if bus_type == 2 or bus_type == 3:
            Vm_combined.append(action[num_gens:][Q_load_index])
            Q_load_index += 1
        elif bus_type == 1:
            Vm_combined.append(PQ_V[PQ_index])
            PQ_index += 1
    Vm_combined = np.array(Vm_combined)
    PQV_theta = np.array(delta)

    voltage_tensor = Vm_combined * np.exp(1j * PQV_theta)  # Complex voltage calculation

    # Real and reactive power injections
    for i in range(num_buses):
        # Power demand at bus
        P_load = state[i]  # Load active power
        Q_load = state[i + num_buses]  # Load reactive power        

        # Generator power outputs
        Pg = np.sum(action[:54][gen_data_[:, 0] == (i + 1)])  # Active power from generators
        Qg = np.sum(PV_Q[gen_data_[:, 0] == (i + 1)])  # Reactive power from generators

        # Voltage calculations
        
        V_conjugate_sum = np.sum(Ybus[i, :] * voltage_tensor)
        if flag:
            if bus_data[i,1] == 3:
                P_injection = 0
                Pg = P_load + np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum) 
            else:
                P_injection = Pg - P_load - np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum)

            if bus_data[i,1] == 1:
                Q_injection = Qg - Q_load + np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum) 
            else:
                Q_injection = 0
                Qg = Q_load - np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum) 

        else:
            P_injection = Pg - P_load - np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum)
            Q_injection = Qg - Q_load + np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum) 
    
        # Update balance
        P_balance += P_injection ** 2
        Q_balance += Q_injection ** 2

        # Calculate generator cost loss
        cost_params = gencost_data[gen_data_[:, 0] == (i + 1)]
        if cost_params.shape[0] > 0:
            a = cost_params[:, 4]  # 二次成本系数
            b = cost_params[:, 5]  # 一次成本系数
            c = cost_params[:, 6]  # 固定成本
            # cost_loss = np.sum(a * (100*action[:54][gen_data_[:, 0] == (i + 1)])**2 + b * 100 * action[:54][gen_data_[:, 0] == (i + 1)] + c)
            # total_cost_loss += cost_loss
            cost_loss = np.sum(a * (100*Pg)**2 + b * 100 * Pg + c)
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
        V_min = 0.94  # Minimum voltage limit
        V_max = 1.06  # Maximum voltage limit
        
        V_magnitude = np.abs(voltage_tensor[i])  # Voltage magnitude
        if V_magnitude > V_max:
            voltage_loss = np.abs(V_magnitude - V_max)  # L1 norm
            total_voltage_loss += voltage_loss
        elif V_magnitude < V_min:
            voltage_loss = np.abs(V_min - V_magnitude)  # L1 norm
            total_voltage_loss += voltage_loss

    Ybus_ = calculate_ybus_numpy(branch_data, num_buses, bus_data)
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

    return (P_balance / num_buses, Q_balance / num_buses, 
            total_cost_loss, total_active_loss / num_gens, 
            total_reactive_loss / num_gens, total_voltage_loss / num_buses, 
            total_line_loss / num_branchs)

# 使用PyTorch进行模型预测和比较
model.eval()

pinn_p_error = 0
pinn_q_error = 0
pinn_cost_error = 0
pinn_active_error = 0
pinn_reactive_error = 0
pinn_voltage_error = 0
pinn_line_error = 0

base_p_error = 0
base_q_error = 0
base_cost_error = 0
base_active_error = 0
base_reactive_error = 0
base_voltage_error = 0
base_line_error = 0

with torch.no_grad():
    ###########
    _action = model.encode(X_con_test)
    _pre_data_pinn = model.pinn_pf(_action, X_con_test)
    q_ = _pre_data_pinn[:,:54].numpy()
    u_ = _pre_data_pinn[:,54:118].numpy()
    delta_ = _pre_data_pinn[:,118:].numpy()
    ###########

    # 转换为numpy用于评估
    X_con_test_np = X_con_test.numpy()
    _action_np = _action.numpy()
    X_in_test_np = X_in_test.numpy()
    X_other_information_test_np = X_other_information_test.numpy()

    for i in range(X_con_test.shape[0]):
        # PINN模型评估
        _temp_p, _temp_q, _temp_cost, _temp_active, _temp_reactive, _temp_voltage, _temp_line = AC_optimal_power_flow_equations_evaluation(
            case118, X_con_test_np[i,:], _action_np[i,:], q_[i,:], u_[i,:], delta_[i,:],flag = True)
        pinn_p_error += np.sqrt(_temp_p)
        pinn_q_error += np.sqrt(_temp_q)
        pinn_cost_error += _temp_cost
        pinn_active_error += _temp_active
        pinn_reactive_error += _temp_reactive
        pinn_voltage_error += _temp_voltage
        pinn_line_error += _temp_line

        # 基准（真实值）评估
        _temp_p, _temp_q, _temp_cost, _temp_active, _temp_reactive, _temp_voltage, _temp_line = AC_optimal_power_flow_equations_evaluation(
            case118, X_con_test_np[i,:], X_in_test_np[i,:], 
            X_other_information_test_np[i,:54], X_other_information_test_np[i,54:118], 
            X_other_information_test_np[i,118:])
        base_p_error += np.sqrt(_temp_p)
        base_q_error += np.sqrt(_temp_q)
        base_cost_error += _temp_cost
        base_active_error += _temp_active
        base_reactive_error += _temp_reactive
        base_voltage_error += _temp_voltage
        base_line_error += _temp_line

_temp_shape = X_con_test_np.shape[0]

logging.info(f'ACOPF Equation Evaluation Results - P Error: {pinn_p_error/_temp_shape:.6f} | Q Error: {pinn_q_error/_temp_shape:.6f} | Cost Error: {pinn_cost_error/_temp_shape:.6f} | Active Power Limit Error: {pinn_active_error/_temp_shape:.6f} | Reactive Power Limit Error: {pinn_reactive_error/_temp_shape:.6f} | Voltage Limit Error: {pinn_voltage_error/_temp_shape:.6f} | Line Limit Error: {pinn_line_error/_temp_shape:.6f}')

print("\n" + "="*80)
print("ACOPF Equation Evaluation Results")
print("="*80)
print(f"\n PINN Model Errors:")
print(f"P Error: {pinn_p_error/_temp_shape:.6f}")
print(f"Q Error: {pinn_q_error/_temp_shape:.6f}")
print(f"Cost Error: {pinn_cost_error/_temp_shape:.6f}")
print(f"Active Power Limit Error: {pinn_active_error/_temp_shape:.6f}")
print(f"Reactive Power Limit Error: {pinn_reactive_error/_temp_shape:.6f}")
print(f"Voltage Limit Error: {pinn_voltage_error/_temp_shape:.6f}")
print(f"Line Limit Error: {pinn_line_error/_temp_shape:.6f}")

print(f"\n Baseline (True Value) Errors:")
print(f"P Error: {base_p_error/_temp_shape:.6f}")
print(f"Q Error: {base_q_error/_temp_shape:.6f}")
print(f"Cost Error: {base_cost_error/_temp_shape:.6f}")
print(f"Active Power Limit Error: {base_active_error/_temp_shape:.6f}")
print(f"Reactive Power Limit Error: {base_reactive_error/_temp_shape:.6f}")
print(f"Voltage Limit Error: {base_voltage_error/_temp_shape:.6f}")
print(f"Line Limit Error: {base_line_error/_temp_shape:.6f}")

# Create comparison table
comparison_data = {
    'Model': ['PINN', 'Baseline'],
    'P Error': [pinn_p_error/_temp_shape, base_p_error/_temp_shape],
    'Q Error': [pinn_q_error/_temp_shape, base_q_error/_temp_shape],
    'Cost Error': [pinn_cost_error/_temp_shape, base_cost_error/_temp_shape],
    'Active Power Limit Error': [pinn_active_error/_temp_shape, base_active_error/_temp_shape],
    'Reactive Power Limit Error': [pinn_reactive_error/_temp_shape, base_reactive_error/_temp_shape],
    'Voltage Limit Error': [pinn_voltage_error/_temp_shape, base_voltage_error/_temp_shape],
    'Line Limit Error': [pinn_line_error/_temp_shape, base_line_error/_temp_shape]
}

comparison_df = pd.DataFrame(comparison_data)
print(f"\n Detailed Comparison Table:")
print(comparison_df)
